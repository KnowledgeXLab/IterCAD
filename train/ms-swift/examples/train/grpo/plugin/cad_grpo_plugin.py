"""
CAD GRPO Plugin
===============
Provides the multi-turn CAD scheduler, Chamfer/format/progress/process
rewards, monitoring rewards, and Geometry-Viable Prefix Masking (GVPM).

Registrations:
  --multi_turn_scheduler cad_scheduler
  --reward_funcs         cad_chamfer cad_format cad_progress cad_process
                         cad_cd_value cad_invalid

Required service:
  export CAD_REWARD_API_URL=http://localhost:8765

Dataset task types:
  "view": inject rendered-image feedback after successful execution.
  "text": inject text-only feedback after successful execution.
"""

import asyncio
import base64
import os
import re
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import requests

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns


_DEBUG_PRINT_COUNT = 0


def _debug_enabled() -> bool:
    return os.environ.get("CAD_DEBUG_REWARD", "0") == "1"


def _debug_max_prints() -> int:
    return int(os.environ.get("CAD_DEBUG_MAX_PRINTS", "12"))


def _is_debug_rank() -> bool:
    return os.environ.get("RANK", "0") == "0"


def _shorten(text: Optional[str], limit: int = 400) -> str:
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _strip_prefixed_empty_think(text: Optional[str]) -> str:
    if not text:
        return ""
    prefixes = (
        "<think>\n\n</think>\n\n",
        "<thinking>\n\n</thinking>\n\n",
    )
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _debug_log(title: str, **payload) -> None:
    global _DEBUG_PRINT_COUNT
    if not (_debug_enabled() and _is_debug_rank()):
        return
    if _DEBUG_PRINT_COUNT >= _debug_max_prints():
        return
    _DEBUG_PRINT_COUNT += 1
    rank = os.environ.get("RANK", "0")
    print(f"[CAD_DEBUG][rank={rank}][{_DEBUG_PRINT_COUNT}] {title}")
    for key, value in payload.items():
        print(f"[CAD_DEBUG] {key}={value}")


def _extract_generated_tail(text: Optional[str]) -> str:
    return _strip_prefixed_empty_think(text)

# Shared prompt and helpers

SYSTEM_PROMPT = (
    "## Role\n"
    "You are a step-by-step CAD reasoning assistant.\n"
    "You are an expert CAD engineer and Python programmer specialized in CadQuery.\n"
    "Given visual, textual, code, and optional feedback inputs, your task is to generate or modify "
    "a 3D CAD model **one reasoning step at a time**.\n\n"
    "## Input Sources:\n"
    "The input may include one or more of the following:\n"
    "1. Technical Drawing Image: orthographic projections such as Front, Top, Side, and ISO views with dimensions.\n"
    "2. Text: modeling instructions, dimensional constraints, or edit requests.\n"
    "3. Existing Code: a CadQuery script that should be preserved or modified when possible.\n\n"
    "## Objective\n"
    "Create or edit a 3D model that satisfies the user request.\n\n"
    "## Output Format (strict)\n"
    "Always start with `<thinking>` and output one of the following:\n\n"
    "1. Structure your `<thinking>` process strictly based on the provided inputs:\n"
    "   - IF NO FEEDBACK IS PROVIDED (e.g., initial generation or completely new instructions):\n"
    "     * Requirement Analysis: Break down visual/textual inputs into CadQuery features.\n"
    "     * Plan: Define origin, workplanes, sketch sequence, booleans, and key dimensions.\n"
    "   - IF FEEDBACK OR ERRORS ARE PROVIDED (e.g., fixing a previous attempt):\n"
    "     * Feedback Analysis: Precisely identify what failed (compilation errors, missing geometry et.al.) "
    "based on the feedback.\n"
    "     * Modification Plan: State the targeted local edits required. Explicitly define what existing code "
    "must be preserved.\n"
    "   - FOR ALL SCENARIOS, always conclude thinking with:\n"
    "     * Precision Check: Verify numerical values and confirm all 2D profiles are closed before extrusion.\n"
    "After `</thinking>`, output exactly one fenced Python block: open with ```python, "
    "then `import cadquery as cq`, your implementation, and assign the final solid to variable `r`, "
    "then close the fence with ```.\n\n"
    "2. If feedback explicitly confirms the 3D model is correct with no remaining issues, briefly state "
    "your assessment inside `<thinking></thinking>`, then output `<DONE>` (no code block).\n\n"
    "## Code Implementation Rules\n"
    "- Language: Python.\n"
    "- Library: `import cadquery as cq`.\n"
    "- Final result must be assigned to variable `r`.\n"
    "- If scaling is needed, define `scale_factor` and multiply dimensions explicitly. Never use `.scale()`.\n"
    "- Ensure all 2D profiles are valid and closed before extrusion when required.\n"
    "- Do not use visualization calls such as `show_object()`.\n"
    "- Preserve correct existing geometry when editing code.\n"
    "- Do not rewrite unrelated parts of the code if only local edits are needed.\n"
)


def _reward_api_url() -> str:
    return os.environ.get("CAD_REWARD_API_URL", "http://localhost:8765")


def _extract_code(text: str) -> Optional[str]:
    """Return the final fenced Python block."""
    matches = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def _has_done(text: str) -> bool:
    return "<DONE>" in text


def _has_thinking(text: str) -> bool:
    """Return whether text contains a complete thinking block."""
    return bool(re.search(r'<thinking>.*?</thinking>', text, re.DOTALL))


def _get_msg_content(msg: dict) -> str:
    """Extract text from string or structured message content."""
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") for c in content if c.get("type") == "text"
        )
    return content


def _cd_to_reward(cd: Optional[float]) -> float:
    """Map Chamfer Distance to a reward in [0, 1]."""
    if cd is None:
        return 0.0
    if cd < 1e-5:
        return 1.0
    if cd > 0.005:
        return 0.0
    slope = (0.005 - 1.0) / (0.005 - 1e-5)
    return max(0.0, 1.0 + (cd - 1e-5) * slope)


def _format_reward_from_messages(messages: list) -> float:
    """
    Validate every assistant turn.

    Each turn needs a thinking block plus either code or ``<DONE>``.
    The first turn cannot stop without attempting code.
    """
    assistant_turns = [
        _get_msg_content(m)
        for m in messages
        if m.get("role") == "assistant"
    ]
    if not assistant_turns:
        return 0.0

    for i, content in enumerate(assistant_turns):
        has_thinking = _has_thinking(content)
        has_code = _extract_code(content) is not None
        has_done = _has_done(content)

        if not has_thinking:
            return 0.0

        if i == 0 and has_done and not has_code:
            return 0.0

        if not has_code and not has_done:
            return 0.0

    return 1.0


def _call_api(code: str, gt_stl_path: Optional[str] = None,
              render: bool = False, sample_points: int = 8192) -> dict:
    """Call the reward API and return execution, CD, and render results."""
    try:
        resp = requests.post(
            f"{_reward_api_url()}/evaluate",
            json={
                "code": code,
                "gt_stl_path": gt_stl_path,
                "render": render,
                "sample_points": sample_points,
            },
            timeout=90,
            proxies={"http": None, "https": None},
        )
        resp.raise_for_status()
        result = resp.json()
        _debug_log(
            "reward_api_ok",
            stl_path=gt_stl_path,
            code_preview=_shorten(code, 300),
            exec_ok=result.get("exec_ok"),
            cd=result.get("cd"),
            exec_msg=_shorten(result.get("exec_msg"), 200),
        )
        return result
    except Exception as exc:
        _debug_log(
            "reward_api_error",
            stl_path=gt_stl_path,
            code_preview=_shorten(code, 300),
            error=_shorten(str(exc), 300),
        )
        return {"exec_ok": False, "exec_msg": str(exc), "cd": None, "render_b64": None}


async def _call_api_async(code: str, gt_stl_path: Optional[str] = None,
                          render: bool = False, sample_points: int = 8192) -> dict:
    """Run a reward API request without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: _call_api(code, gt_stl_path, render, sample_points),
    )


# Cache identical reward requests across reward functions.

_api_cache: Dict = {}
_api_cache_lock = threading.Lock()
_API_CACHE_MAX = 2000


def _call_api_cached(code: str, gt_stl_path: Optional[str], sample_points: int) -> dict:
    """Cache reward API results by code and STL path."""
    key = (code, gt_stl_path)
    with _api_cache_lock:
        if key in _api_cache:
            return _api_cache[key]
    result = _call_api(code, gt_stl_path=gt_stl_path, sample_points=sample_points)
    with _api_cache_lock:
        if len(_api_cache) >= _API_CACHE_MAX:
            oldest = next(iter(_api_cache))
            del _api_cache[oldest]
        _api_cache[key] = result
    return result


def _save_render_to_tmpfile(render_b64: str) -> Optional[str]:
    """Decode a rendered image into a temporary file."""
    try:
        img_bytes = base64.b64decode(render_b64)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False, prefix="cad_rl_render_"
        )
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name
    except Exception as exc:
        _debug_log("render_save_error", error=_shorten(str(exc), 200))
        return None


# Multi-turn scheduler

def _mask_traj_enabled() -> bool:
    """Geometry-Viable Prefix Masking (GVPM); enabled when CAD_MASK_TRAJ=1."""
    return os.environ.get("CAD_MASK_TRAJ", "0") == "1"


def _mask_traj_consecutive_k() -> int:
    """Rule 1: consecutive execution failures before prefix is masked (default: 2)."""
    return int(os.environ.get("CAD_MASK_TRAJ_K", "2"))


def _mask_traj_stagnant_threshold() -> float:
    """Rule 2: CD above this value with no improvement → geometry stall (default: 0.01)."""
    return float(os.environ.get("CAD_MASK_TRAJ_STAGNANT_CD", "0.01"))


class CADMultiTurnScheduler(MultiTurnScheduler):
    """
    Run iterative CAD generation, execution, feedback, and correction.

    View tasks receive rendered feedback; text tasks receive text feedback.
    The scheduler stops on ``<DONE>``, missing code, or the parent limits.
    GVPM masks execution cascades and stalled geometry when enabled.
    """

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        content = response_choice.message.content or ""
        if _has_done(content):
            return True
        if _extract_code(content) is None:
            return True
        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request, response_choice, current_turn: int) -> Dict:
        """
        Execute one colocated rollout step.

        Batch-level concurrency is managed by the rollout implementation.
        The trainer patch applies GVPM masking and advantage clamping.
        """
        content = response_choice.message.content or ""
        code = _extract_code(content)

        task_type = (infer_request.data_dict or {}).get("task_type", "view")
        need_render = (task_type == "view")

        ret_base = {"infer_request": infer_request}

        if code is None:
            feedback = (
                "Please output valid Python code inside a ```python``` block. "
                "Do NOT output <DONE> before providing code."
            )
            infer_request.messages.append({"role": "user", "content": feedback})
            return ret_base

        result = _call_api(code, render=need_render)

        if not result["exec_ok"]:
            err = result.get("exec_msg", "Unknown error")
            feedback = (
                f"[Turn Feedback] Runtime Error:\n"
                f"```\n{err}\n```\n\n"
                "Feedback Analysis: Precisely identify what failed.\n"
                "Modification Plan: State the targeted local edits required. "
                "Explicitly define what existing code must be preserved.\n\n"
                "Please fix the code."
            )
            infer_request.messages.append({"role": "user", "content": feedback})
            return ret_base

        render_path = None
        if need_render:
            render_b64 = result.get("render_b64")
            render_path = _save_render_to_tmpfile(render_b64) if render_b64 else None

        if render_path:
            infer_request.images.append(render_path)
            feedback = (
                "Your generated model views from the previous turn are shown below.\n"
                "Compare them with the original design request provided at the beginning of this conversation.\n\n"
                "If the model is correct and complete, output <DONE>.\n"
                "Otherwise, briefly analyze the remaining issues in <thinking></thinking> tags "
                "and provide corrected code.\n"
                "<image>"
            )
        elif need_render:
            feedback = (
                "Your code executed successfully, "
                "but the 3D model could not be rendered (possibly empty geometry).\n"
                "Please check that the model produces valid non-empty geometry and fix the code."
            )
        else:
            feedback = (
                "Your code from the previous turn executed successfully.\n"
                "Check whether it fully and correctly applies the editing instruction.\n\n"
                "If it is correct and complete, output <DONE>.\n"
                "Otherwise, briefly analyze the remaining issues in <thinking></thinking> tags "
                "and provide corrected code."
            )

        infer_request.messages.append({"role": "user", "content": feedback})
        return ret_base


multi_turns["cad_scheduler"] = CADMultiTurnScheduler


# ---------------------------------------------------------------------------
# GVPM — Geometry-Viable Prefix Masking (CAD_MASK_TRAJ=1)
# ---------------------------------------------------------------------------
# Paper-aligned (2 rules, no cad_process):
#   Rule 1 — Execution cascade: K consecutive CadQuery Runtime Errors.
#   Rule 2 — Geometry stall: ≥2 valid CDs, no strict CD decrease, latest CD > η.
#   Prefix step f = min(f_exec, f_stall); mask assistant tokens for turns ≥ f.
#   Advantage clamp: if Rule 1 or Rule 2 fires, Â = max(A, 0) (one-sided).
# ---------------------------------------------------------------------------


def _collect_turn_cd_trajectory(
    messages: list,
    stl_path: Optional[str],
    sample_points: int,
) -> List[Optional[float]]:
    """Per assistant turn: CD if exec_ok, else None."""
    if not stl_path:
        return []
    trajectory: List[Optional[float]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        code = _extract_code(_get_msg_content(msg))
        if not code:
            trajectory.append(None)
            continue
        result = _call_api_cached(code, stl_path, sample_points)
        if result["exec_ok"] and result.get("cd") is not None:
            trajectory.append(float(result["cd"]))
        else:
            trajectory.append(None)
    return trajectory


def _detect_exec_fatal_step(messages: list, fatal_k: int) -> Optional[int]:
    """
    Rule 1 — Execution cascade.

    Returns 1-based index of the first assistant turn in a run of K consecutive
    execution failures (user feedback contains "Runtime Error").
    """
    # Build a list of (assistant_turn_index, is_error) pairs
    consecutive_errors = 0
    assistant_turn_idx = 0

    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        assistant_turn_idx += 1

        # Check if the next user message contains "Runtime Error"
        next_user_msg = ""
        for j in range(i + 1, len(messages)):
            if messages[j].get("role") == "user":
                next_user_msg = _get_msg_content(messages[j])
                break

        if "Runtime Error" in next_user_msg:
            consecutive_errors += 1
            if consecutive_errors >= fatal_k:
                # Fatal! Return the first turn of the cascade
                return assistant_turn_idx - fatal_k + 1
        else:
            consecutive_errors = 0

    return None


def _detect_geometry_stall_step(
    messages: list,
    stl_path: Optional[str],
    threshold: float,
    sample_points: int,
) -> Optional[int]:
    """
    Rule 2 — Geometry stall (earliest mask step).

    After assistant turn t, if there are ≥2 valid CDs, none strictly decreased,
    and the latest CD > threshold, return t+1 (mask from the next turn onward).
    """
    turn_cds = _collect_turn_cd_trajectory(messages, stl_path, sample_points)
    cd_history: List[float] = []
    for turn_idx, cd in enumerate(turn_cds, start=1):
        if cd is None:
            continue
        cd_history.append(cd)
        if len(cd_history) < 2:
            continue
        if cd_history[-1] <= threshold:
            continue
        if any(cd_history[i] < cd_history[i - 1] for i in range(1, len(cd_history))):
            continue
        return turn_idx + 1
    return None


def _is_geometry_stall_trajectory(
    messages: list,
    stl_path: Optional[str],
    threshold: float,
    sample_points: int,
) -> bool:
    """Rule 2 at trajectory level (for clamp when mask step is past last turn)."""
    cds = [cd for cd in _collect_turn_cd_trajectory(messages, stl_path, sample_points) if cd is not None]
    if len(cds) < 2:
        return False
    if cds[-1] <= threshold:
        return False
    return all(cds[i] >= cds[i - 1] for i in range(1, len(cds)))


def _gvpm_prefix_step(
    messages: list,
    stl_path: Optional[str],
    fatal_k: int,
    stagnant_threshold: float,
    sample_points: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (f, f_exec, f_stall) with f = min(non-null steps)."""
    f_exec = _detect_exec_fatal_step(messages, fatal_k)
    f_stall = _detect_geometry_stall_step(messages, stl_path, stagnant_threshold, sample_points)
    steps = [s for s in (f_exec, f_stall) if s is not None]
    if not steps:
        return None, f_exec, f_stall
    return min(steps), f_exec, f_stall


def _apply_prefix_mask_to_response_loss_mask(
    response_loss_mask: List[List[int]],
    prefix_step_index: int,
) -> List[List[int]]:
    """
    Zero out response_loss_mask for all turns >= prefix_step_index (1-based).

    response_loss_mask is a list of lists, one per assistant turn.
    """
    result = []
    for turn_idx, turn_mask in enumerate(response_loss_mask):
        # turn_idx is 0-based; fatal_step_index is 1-based
        if (turn_idx + 1) >= prefix_step_index:
            result.append([0] * len(turn_mask))
        else:
            result.append(turn_mask)
    return result


def _gvpm_postprocess(inputs: list) -> list:
    """
    Apply GVPM token masking and set rollout_infos for advantage clamping.

    f = min(f_exec, f_stall). Mask turns t >= f when f is within num assistant turns.
    Clamp when Rule 1 fires, Rule 2 fires (mask step or trajectory-level stall).
    """
    fatal_k = _mask_traj_consecutive_k()
    stagnant_threshold = _mask_traj_stagnant_threshold()
    sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))
    exec_count = 0
    stall_count = 0
    masked_count = 0

    for inp in inputs:
        messages = inp.get("messages", [])
        if not messages:
            continue

        if "rollout_infos" not in inp:
            inp["rollout_infos"] = {}

        stl_path = inp.get("gt_stl_path", None)
        f, f_exec, f_stall = _gvpm_prefix_step(
            messages, stl_path, fatal_k, stagnant_threshold, sample_points
        )
        is_stall_traj = _is_geometry_stall_trajectory(
            messages, stl_path, stagnant_threshold, sample_points
        )

        num_turns = len(inp.get("response_loss_mask") or [])
        # Cap f to num_turns: when stall/exec-cascade is confirmed on the last
        # turn, f = num_turns+1 which would skip masking entirely.  Capping to
        # num_turns masks (at least) the final turn — the one that confirmed
        # the problem — which is the desired behaviour.
        if f is not None and num_turns > 0 and f > num_turns:
            f = num_turns
        apply_mask = f is not None and num_turns > 0 and f <= num_turns

        if f_exec is not None:
            exec_count += 1
        if f_stall is not None or is_stall_traj:
            stall_count += 1

        if apply_mask:
            masked_count += 1
            inp["response_loss_mask"] = _apply_prefix_mask_to_response_loss_mask(
                inp["response_loss_mask"], f
            )

        should_clamp = (f_exec is not None) or (f_stall is not None) or is_stall_traj
        inp["rollout_infos"].update(
            {
                "gvpm_prefix_step": f,
                "gvpm_f_exec": f_exec,
                "gvpm_f_stall": f_stall,
                "gvpm_is_stall_trajectory": is_stall_traj,
                "gvpm_should_clamp": should_clamp,
                # Legacy keys for any downstream logging
                "is_fatal": f_exec is not None,
                "is_stagnant": is_stall_traj or f_stall is not None,
                "fatal_step_index": f,
            }
        )

    if os.environ.get("RANK", "0") == "0" and (exec_count > 0 or stall_count > 0):
        print(
            f"[CAD_GVPM] exec_cascade={exec_count} geometry_stall={stall_count} "
            f"masked={masked_count} / {len(inputs)} (K={fatal_k}, η={stagnant_threshold})",
            flush=True,
        )

    return inputs


# ---------------------------------------------------------------------------
# One-sided advantage clamping (GRPO): Â = max(A, 0) when GVPM triggers
# ---------------------------------------------------------------------------

def _install_gvpm_masking(trainer_cls):
    """
    Monkey-patch GRPO trainer: GVPM prefix mask + one-sided advantage clamp.

    Only when CAD_MASK_TRAJ=1. Does not use cad_process.
    """
    original_generate_completions = trainer_cls._generate_completions

    def _patched_generate_completions(self, inputs):
        result = original_generate_completions(self, inputs)
        _gvpm_postprocess(result)
        # _postprocess_rollout_outputs does deepcopy(input_data), so the
        # returned `result` contains NEW dict objects.  The caller
        # (_generate_and_score_completions) reassigns its local `inputs`
        # to this new list, but the `inputs` parameter in the *outer*
        # _patched_generate_and_score still points to the ORIGINAL dicts
        # which lack rollout_infos.  Stash the processed list so that the
        # advantage-clamping patch can find it.
        self._gvpm_processed_inputs = result
        return result

    trainer_cls._generate_completions = _patched_generate_completions

    original_generate_and_score = trainer_cls._generate_and_score_completions

    def _patched_generate_and_score(self, inputs):
        batch_encoded_inputs = original_generate_and_score(self, inputs)

        # Use the processed inputs that _patched_generate_completions saved.
        # These are the deepcopy'd dicts that actually carry rollout_infos
        # (including gvpm_should_clamp).  The `inputs` parameter here still
        # points to the *original* pre-deepcopy dicts which do NOT have
        # rollout_infos — that was the root cause of clamping never firing.
        processed = getattr(self, "_gvpm_processed_inputs", None) or inputs
        self._gvpm_processed_inputs = None  # avoid stale references

        clamped_count = 0
        total_clampable = 0
        gas_chunks = self.split_by_mini_batches(processed)
        for batch, batch_encoded in zip(gas_chunks, batch_encoded_inputs):
            advantages = batch_encoded.get("advantages")
            if advantages is None:
                continue
            for i, data in enumerate(batch):
                infos = data.get("rollout_infos") or {}
                should_clamp = infos.get("gvpm_should_clamp", False)
                if should_clamp:
                    total_clampable += 1
                    if advantages[i] < 0:
                        advantages[i] = 0.0
                        clamped_count += 1

        if os.environ.get("RANK", "0") == "0" and total_clampable > 0:
            print(
                f"[CAD_GVPM] Clamped {clamped_count}/{total_clampable} negative advantages (Â=max(A,0))",
                flush=True,
            )

        # --- Log GVPM metrics so they appear in logging.jsonl / tensorboard ---
        if hasattr(self, "_metrics"):
            mode = "train" if self.model.training else "eval"
            n = len(processed)
            if n > 0:
                mask_ratio_sum = 0.0
                clamp_flag_sum = 0.0
                for data in processed:
                    infos = data.get("rollout_infos") or {}
                    rlm = data.get("response_loss_mask") or []
                    if rlm:
                        total_tokens = sum(len(t) for t in rlm)
                        masked_tokens = sum(1 for t in rlm for v in t if v == 0)
                        mask_ratio_sum += (
                            masked_tokens / total_tokens if total_tokens > 0 else 0.0
                        )
                    clamp_flag_sum += (
                        1.0 if infos.get("gvpm_should_clamp", False) else 0.0
                    )
                self._metrics[mode]["gvpm/mask_ratio"].append(mask_ratio_sum / n)
                self._metrics[mode]["gvpm/clamp_rate"].append(clamp_flag_sum / n)

        return batch_encoded_inputs

    trainer_cls._generate_and_score_completions = _patched_generate_and_score

    if os.environ.get("RANK", "0") == "0":
        print(
            f"[CAD_GVPM] Geometry-Viable Prefix Masking installed "
            f"(K={_mask_traj_consecutive_k()}, η={_mask_traj_stagnant_threshold()})",
            flush=True,
        )


def _install_fatal_aware_masking(trainer_cls):
    """Backward-compatible alias."""
    _install_gvpm_masking(trainer_cls)


# Auto-install when this plugin is loaded with CAD_MASK_TRAJ=1
if _mask_traj_enabled():
    try:
        from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

        _install_gvpm_masking(GRPOTrainer)
    except Exception as _exc:
        if os.environ.get("RANK", "0") == "0":
            print(
                f"[CAD_GVPM] WARNING: Failed to install GVPM monkey-patch: {_exc}",
                flush=True,
            )
else:
    if os.environ.get("RANK", "0") == "0":
        print(
            f"[CAD_GVPM] DISABLED (CAD_MASK_TRAJ={os.environ.get('CAD_MASK_TRAJ', 'unset')})",
            flush=True,
        )


# Reward functions

class CADChamferReward(ORM):
    """
    Return a piecewise-linear Chamfer Distance reward.

    Missing code or execution failures receive zero. Format and progress
    rewards are provided separately.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))

    @staticmethod
    def _last_code_from_messages(messages: list) -> Optional[str]:
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            code = _extract_code(_get_msg_content(msg))
            if code:
                return code
        return None

    def _score(self, messages: list, stl_path: str) -> float:
        code = self._last_code_from_messages(messages)
        if not code:
            _debug_log(
                "no_code_block",
                stl_path=stl_path,
                stl_exists=os.path.exists(stl_path) if stl_path else False,
            )
            return 0.0

        result = _call_api_cached(code, gt_stl_path=stl_path, sample_points=self.sample_points)
        if not result["exec_ok"]:
            return 0.0

        return _cd_to_reward(result.get("cd"))

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        """
        Args:
            completions: Final completion for each batch item.
            messages: Complete conversation history for each item.
            gt_stl_path: Ground-truth STL path for each item.
        """
        n = len(completions)
        rewards = [None] * n

        def _score_one(args):
            idx, completion, msg_list, stl_path = args
            _debug_log(
                "reward_input",
                sample_idx=idx,
                stl_path=stl_path,
                stl_exists=os.path.exists(stl_path) if stl_path else False,
                completion_preview=_shorten(completion, 500),
                generated_tail_preview=_shorten(_extract_generated_tail(completion), 300),
                has_code_block=bool(_extract_code(completion)),
            )
            try:
                return idx, self._score(msg_list, stl_path)
            except Exception:
                _debug_log(
                    "reward_fallback_exception",
                    sample_idx=idx,
                    stl_path=stl_path,
                    completion_preview=_shorten(completion, 300),
                )
                return idx, -0.3

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [
                executor.submit(_score_one, (idx, completion, msg_list, stl_path))
                for idx, (completion, msg_list, stl_path) in enumerate(
                    zip(completions, messages, gt_stl_path)
                )
            ]
            for future in as_completed(futures):
                idx, r = future.result()
                rewards[idx] = r
        return rewards


orms["cad_chamfer"] = CADChamferReward


# Raw CD monitoring reward

class CADCDValueReward(ORM):
    """
    Return raw Chamfer Distance, or NaN for invalid samples.

    Use weight 0.0 to log ``rewards/cad_cd_value/mean`` without affecting
    gradients.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        n = len(completions)
        rewards = [float("nan")] * n

        def _get_cd(args):
            idx, msg_list, stl_path = args
            try:
                code = CADChamferReward._last_code_from_messages(msg_list)
                if not code:
                    return idx, float("nan")
                result = _call_api_cached(code, stl_path, self.sample_points)
                if not result["exec_ok"]:
                    return idx, float("nan")
                cd = result.get("cd")
                return idx, cd if cd is not None else float("nan")
            except Exception:
                return idx, float("nan")

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [
                executor.submit(_get_cd, (idx, msg_list, stl_path))
                for idx, (msg_list, stl_path) in enumerate(zip(messages, gt_stl_path))
            ]
            for future in as_completed(futures):
                idx, val = future.result()
                rewards[idx] = val

        return rewards


orms["cad_cd_value"] = CADCDValueReward


# Invalid-sample monitoring reward

class CADInvalidReward(ORM):
    """
    Return 1.0 for execution failures or missing CD, otherwise 0.0.

    Use weight 0.0 to log ``rewards/cad_invalid/mean`` without affecting
    gradients.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        n = len(completions)
        rewards = [1.0] * n

        def _get_invalid(args):
            idx, msg_list, stl_path = args
            try:
                code = CADChamferReward._last_code_from_messages(msg_list)
                if not code:
                    return idx, 1.0
                result = _call_api_cached(code, stl_path, self.sample_points)
                if not result["exec_ok"] or result.get("cd") is None:
                    return idx, 1.0
                return idx, 0.0
            except Exception:
                return idx, 1.0

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [
                executor.submit(_get_invalid, (idx, msg_list, stl_path))
                for idx, (msg_list, stl_path) in enumerate(zip(messages, gt_stl_path))
            ]
            for future in as_completed(futures):
                idx, val = future.result()
                rewards[idx] = val

        return rewards


orms["cad_invalid"] = CADInvalidReward


# Format reward

class CADFormatReward(ORM):
    """
    Return 1.0 when every assistant turn follows the required format.

    Every turn needs a thinking block and either code or ``<DONE>``. The first
    turn must include a code attempt.
    """

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        rewards = []
        for msg_list in messages:
            try:
                rewards.append(_format_reward_from_messages(msg_list))
            except Exception:
                rewards.append(0.0)
        return rewards


orms["cad_format"] = CADFormatReward


# Multi-turn progress reward

class CADProgressReward(ORM):
    """
    Reward genuine improvement from the first valid turn to the final turn.

    At least two code turns are required. The first must execute successfully,
    and the final CD must improve and pass the configured quality threshold.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))
        self.cd_threshold = float(os.environ.get("CAD_PROGRESS_CD_THRESHOLD", "0.3"))
        self.bonus = float(os.environ.get("CAD_PROGRESS_BONUS", "0.1"))

    def _score(self, messages: list, stl_path: str) -> float:
        codes = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            code = _extract_code(_get_msg_content(msg))
            if code:
                codes.append(code)

        if len(codes) < 2:
            return 0.0

        r0 = _call_api_cached(codes[0], stl_path, self.sample_points)
        if not r0["exec_ok"] or r0.get("cd") is None:
            return 0.0

        cd_first = r0["cd"]

        r_final = _call_api_cached(codes[-1], stl_path, self.sample_points)
        if not r_final["exec_ok"] or r_final.get("cd") is None:
            return 0.0

        cd_final = r_final["cd"]

        if cd_final >= cd_first or cd_final >= self.cd_threshold:
            return 0.0

        return self.bonus

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        n = len(completions)
        rewards = [0.0] * n

        def _score_one(args):
            idx, msg_list, stl_path = args
            try:
                return idx, self._score(msg_list, stl_path)
            except Exception:
                return idx, 0.0

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [
                executor.submit(_score_one, (idx, msg_list, stl_path))
                for idx, (msg_list, stl_path) in enumerate(zip(messages, gt_stl_path))
            ]
            for future in as_completed(futures):
                idx, r = future.result()
                rewards[idx] = r

        return rewards


orms["cad_progress"] = CADProgressReward

# Dense process reward

class CADProcessReward(ORM):
    """
    Return the fraction of comparable consecutive turns that improve CD.

    Failed executions remain in the trajectory as invalid points and do not
    form comparable pairs.
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.sample_points = int(os.environ.get("CAD_SAMPLE_POINTS", "8192"))

    def _score(self, messages: list, stl_path: str) -> float:
        """Score one trajectory."""
        cd_trajectory = []
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            code = _extract_code(_get_msg_content(msg))
            if not code:
                continue
            result = _call_api_cached(code, stl_path, self.sample_points)
            if result["exec_ok"] and result.get("cd") is not None:
                cd_trajectory.append(result["cd"])
            else:
                cd_trajectory.append(None)

        improvements = 0
        valid_pairs = 0
        for t in range(1, len(cd_trajectory)):
            if cd_trajectory[t] is not None and cd_trajectory[t - 1] is not None:
                valid_pairs += 1
                if cd_trajectory[t] < cd_trajectory[t - 1]:
                    improvements += 1

        if valid_pairs == 0:
            return 0.0

        return improvements / valid_pairs

    def __call__(self, completions, messages, gt_stl_path, **kwargs) -> List[float]:
        n = len(completions)
        rewards = [0.0] * n

        def _score_one(args):
            idx, msg_list, stl_path = args
            try:
                return idx, self._score(msg_list, stl_path)
            except Exception:
                return idx, 0.0

        with ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            futures = [
                executor.submit(_score_one, (idx, msg_list, stl_path))
                for idx, (msg_list, stl_path) in enumerate(zip(messages, gt_stl_path))
            ]
            for future in as_completed(futures):
                idx, r = future.result()
                rewards[idx] = r

        return rewards


orms["cad_process"] = CADProcessReward
