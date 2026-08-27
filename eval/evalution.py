#!/usr/bin/env python3
"""
Unified Eval Pipeline for IterCAD-Draw and IterCAD-Edit tasks.
- IterCAD-Edit: text instruction editing only (no visual edit)
- Agentic multi-turn generation with execution feedback
- supports vLLM extra_body parameters
"""
import os
import json
import base64
import io
import re
import sys
import time
import math
import numpy as np
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
import multiprocessing
import argparse
from openai import OpenAI
from PIL import Image

from code_executor import execute_code, render_views, compute_cd

# Repo root (parent of eval/), used for portable default benchmark paths.
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_EVAL_DIR)
TASK_ITERCAD_DRAW = "IterCAD-Draw"
TASK_ITERCAD_EDIT = "IterCAD-Edit"
TASK_LEGACY_ALIASES = {
    TASK_ITERCAD_DRAW: ["view2code"],
    TASK_ITERCAD_EDIT: ["edit"],
}
DEFAULT_ITERCAD_DRAW_JSONL = os.path.join(
    REPO_ROOT, "IterCAD_data/benchmark/IterCAD-Draw_1k/IterCAD-Draw_1k.jsonl"
)
DEFAULT_ITERCAD_EDIT_JSONL = os.path.join(
    REPO_ROOT, "IterCAD_data/benchmark/IterCAD-Edit_200/IterCAD-Edit_200.jsonl"
)

SAMPLE_PATH_KEYS = (
    "gt_view_path",
    "gt_stl_path",
    "gt_stl_normalized_path",
    "code_path",
    "original_code_path",
    "input_code_path",
    "target_code_path",
    "input_stl_path",
    "target_stl_path",
    "input_step_path",
    "target_step_path",
)

# ==========================================
# 1. Configuration
# ==========================================

# Aligned with cad_grpo_plugin.SYSTEM_PROMPT (training multi-turn scheduler).
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
    "   </thinking>\n"
    "   ```python\n"
    "   import cadquery as cq\n"
    "   # ... your code ...\n"
    "   r = ...\n"
    "   ```\n"
    "2. If feedback explicitly confirms the 3D model is correct with no remaining issues, briefly state your assessment in `<thinking> </thinking>`, then output `<DONE>`\n"

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

DEFAULT_CONFIG = {
    "generator_api": os.getenv("GENERATOR_API") or "http://127.0.0.1:8000/v1",
    "generator_api_key": os.getenv("OPENAI_API_KEY") or "EMPTY",
    "gen_model_name": os.getenv("GEN_MODEL") or None,
    "max_turns": 5,
    "sample_points": 8192,
    "max_workers": 4,
    "queue_size": 8,
    "auc_tr_min_cd": 1e-5,
    "auc_tr_max_cd": 1e-1,
    "auc_tr_num_points": 401,
    "temp_dir": "/tmp/cadquery_eval",
    "output_dir": "results_unified",
    "system_prompt": SYSTEM_PROMPT,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    "pass_k": 2,
}

# ==========================================
# 2. Utilities
# ==========================================

def _empty_assistant_error_detail(choice, resp):
    """Build a diagnostic message when the API returns a choice without usable assistant text."""
    parts = []
    msg = getattr(choice, "message", None)
    finish = getattr(choice, "finish_reason", None)
    if finish is not None:
        parts.append(f"finish_reason={finish!r}")
    if msg is not None:
        refusal = getattr(msg, "refusal", None)
        if refusal:
            parts.append(f"refusal={refusal!r}")
        tc = getattr(msg, "tool_calls", None)
        if tc:
            parts.append(f"tool_calls={len(tc)}")
        c = getattr(msg, "content", None)
        if c is None:
            parts.append("content=None")
        elif isinstance(c, str) and not c.strip():
            parts.append("content_empty_or_whitespace")
        elif isinstance(c, list):
            parts.append(f"content_list_len={len(c)}")
    rid = getattr(resp, "id", None)
    if rid:
        parts.append(f"response_id={rid!r}")
    if not parts:
        parts.append("no_detail")
    return "empty_assistant_message: " + "; ".join(parts)


def _normalize_message_content(content):
    """Flatten list-shaped message content from some gateways into a single string."""
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(part.get("text") or "")
            else:
                t = getattr(part, "text", None)
                if t:
                    texts.append(t)
        return "".join(texts) if texts else None
    return str(content) if content is not None else None


class GeneratorClient:
    """Generator API client."""

    def __init__(self, config):
        self.client = OpenAI(
            api_key=config.get('generator_api_key') or "EMPTY",
            base_url=config['generator_api']
        )
        self.model_name = config['gen_model_name']
        self.system_prompt = config['system_prompt']
        self.extra_body = config.get('extra_body')
        self.max_tokens = config.get('max_tokens', 40960)

    @staticmethod
    def encode_image(image_path, resize_to=None):
        if not image_path or not os.path.exists(image_path):
            return None
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if resize_to:
                img = img.resize(resize_to, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            fmt = 'PNG' if image_path.lower().endswith('.png') else 'JPEG'
            img.save(buf, format=fmt)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            mime = f"image/{fmt.lower()}"
            return f"data:{mime};base64,{b64}"

    def chat(self, messages, temperature=0.7, timeout=120):
        try:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
            }
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            try:
                resp = self.client.chat.completions.create(timeout=timeout, **kwargs)
            except TypeError:
                resp = self.client.chat.completions.create(**kwargs)

            if not resp or not resp.choices:
                return None, "No valid choices in response"
            choice = resp.choices[0]
            msg = getattr(choice, "message", None)
            raw = getattr(msg, "content", None) if msg is not None else None
            content = _normalize_message_content(raw)
            if content is None or (isinstance(content, str) and not content.strip()):
                err_detail = _empty_assistant_error_detail(choice, resp)
                print(f"[WARN][GeneratorClient] {err_detail}", flush=True)
                return None, err_detail
            return content, None
        except Exception as e:
            traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
            print(f"[WARN][GeneratorClient] {err}", flush=True)
            return None, err


def resolve_data_path(path, base_dir):
    """Resolve a relative path in jsonl to an absolute path."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def normalize_sample_paths(sample, base_dir):
    """Resolve all data-file path fields in a sample."""
    for key in SAMPLE_PATH_KEYS:
        if key in sample and sample[key]:
            sample[key] = resolve_data_path(sample[key], base_dir)
    return sample


def read_code_file(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def export_to_stl(cq_obj, stl_path):
    import cadquery as cq
    try:
        cq.exporters.export(cq_obj, stl_path, exportType="STL")
        return os.path.exists(stl_path)
    except Exception as e:
        print(f"[WARN] STL export failed: {e}")
        return False


def execute_gt_code(gt_code_path, temp_dir, uid):
    code = read_code_file(gt_code_path)
    if not code:
        return None
    exec_ok, exec_msg, cq_obj = execute_code(code)
    if not exec_ok:
        print(f"[WARN] GT code exec failed for {uid}: {exec_msg}")
        return None
    stl_path = os.path.join(temp_dir, f"{uid}_gt.stl")
    return stl_path if export_to_stl(cq_obj, stl_path) else None


def get_uid(sample):
    """Extract unique uid; replace slashes with underscores."""
    task = sample.get('_task_type', TASK_ITERCAD_DRAW)
    base_uid = sample.get('uid', sample.get('source_file', 'unknown'))
    if task in (TASK_ITERCAD_EDIT, 'edit'):
        return f"{base_uid}_v{sample.get('variant_idx', 0)}".replace('/', '_')
    return base_uid.replace('/', '_')


def extract_code(response):
    """Extract Python code and DONE marker from model response (agentic mode)."""
    has_done = '<DONE>' in response
    # Use the last ```python``` block, same as training.
    matches = re.findall(r"```python(.*?)```", response, re.DOTALL)
    code = matches[-1].strip() if matches else None
    return code, has_done


def try_execute_and_eval(code_str, cq_obj_holder, temp_dir, uid, turn, gt_stl_path, sample_points, need_render=False):
    """Execute code, compute CD, optionally render views. Returns (turn_data, cq_obj)."""
    exec_ok, exec_msg, cq_obj = execute_code(code_str)
    result = {'exec_success': exec_ok, 'exec_message': exec_msg}

    if not exec_ok:
        return result, None

    # Optional: render multi-view image (IterCAD-Draw only)
    gen_view_path = None
    if need_render:
        gen_view_path = os.path.join(temp_dir, f"{uid}_turn{turn}_view.png")
        render_ok = render_views(cq_obj, gen_view_path)
        if not render_ok:
            result['status'] = 'render_error'
            return result, None
        result['generated_view_path'] = gen_view_path

    # Compute Chamfer Distance
    if gt_stl_path:
        gen_stl_path = os.path.join(temp_dir, f"{uid}_turn{turn}_gen.stl")
        if export_to_stl(cq_obj, gen_stl_path):
            cd_val = compute_cd(cq_obj, gt_stl_path, sample_points)
            if cd_val is not None:
                result['chamfer_distance'] = cd_val

    return result, cq_obj


# ==========================================
# 3. IterCAD-Draw Worker
# ==========================================

def process_itercad_draw(sample_data, config):
    """Drawing image -> CadQuery code."""
    uid = get_uid(sample_data)
    gen_client = GeneratorClient(config)

    gt_img_url = gen_client.encode_image(sample_data['gt_view_path'])
    if not gt_img_url:
        return {'status': 'skip', 'reason': 'img_load_error', 'uid': uid, 'task_type': TASK_ITERCAD_DRAW}

    temp_dir = config['temp_dir']
    os.makedirs(temp_dir, exist_ok=True)

    gt_stl_path = sample_data.get('gt_stl_normalized_path', '')
    sample_points = config.get('sample_points', 8192)

    # Build initial messages (encode image once; later turns reuse conversation history)
    sys_msg = {"role": "system", "content": gen_client.system_prompt}
    initial_user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the object in the image and convert it to CadQuery code."},
            {"type": "image_url", "image_url": {"url": gt_img_url}}
        ]
    }

    # Conversation history: sys + initial_user, then append assistant/user turns
    conversation = [sys_msg, initial_user_msg]
    all_turns_data = []
    generated_view_paths = []
    final_cd = -1.0
    last_valid_code = None

    for turn in range(config['max_turns']):
        resp, error = gen_client.chat(conversation)
        if not resp:
            all_turns_data.append({'turn': turn, 'status': 'api_error', 'error': error})
            continue

        code_str, has_done = extract_code(resp)
        turn_data = {'turn': turn, 'response': resp, 'code': code_str}

        # Model returned DONE after valid code already exists
        if not code_str and has_done and last_valid_code is not None:
            turn_data['status'] = 'done_by_model'
            all_turns_data.append(turn_data)
            break

        if not code_str:
            turn_data['status'] = 'no_code'
            all_turns_data.append(turn_data)
            # Append to history without re-sending the image
            conversation.append({"role": "assistant", "content": resp})
            if turn == 0 and has_done and last_valid_code is None:
                user_msg = (
                    "Self-reflection: you output <DONE> before any code was executed. "
                    "In <thinking>, briefly re-check the drawing vs. your plan, then output one ```python``` block (variable r). "
                    "Do not output <DONE> until a later turn after review."
                )
            else:
                user_msg = (
                    "Please output valid python code inside ```python``` block. "
                    "Do NOT output <DONE> before generating code."
                )
            conversation.append({"role": "user", "content": user_msg})
            continue

        # Execute code and evaluate
        exec_result, cq_obj = try_execute_and_eval(
            code_str, None, temp_dir, uid, turn, gt_stl_path, sample_points, need_render=True)
        turn_data.update(exec_result)

        if not exec_result.get('exec_success'):
            turn_data['status'] = 'exec_error'
            all_turns_data.append(turn_data)
            conversation.append({"role": "assistant", "content": resp})
            err = exec_result.get('exec_message', 'Unknown error')
            conversation.append({"role": "user", "content": (
                f"[Turn Feedback] Runtime Error:\n"
                f"```\n{err}\n```\n\n"
                "Feedback Analysis: Precisely identify what failed.\n"
                "Modification Plan: State the targeted local edits required. "
                "Explicitly define what existing code must be preserved.\n\n"
                "Please fix the code."
            )})
            continue

        last_valid_code = code_str
        if 'chamfer_distance' in exec_result:
            final_cd = exec_result['chamfer_distance']
        if 'generated_view_path' in exec_result:
            generated_view_paths.append(exec_result['generated_view_path'])

        if exec_result.get('status') == 'render_error':
            turn_data['status'] = 'render_error'
        else:
            turn_data['status'] = 'exec_success'
        all_turns_data.append(turn_data)

        # Same as training eval: DONE after a successful turn is accepted only when turn > 0
        if has_done and turn > 0:
            break

        # Visual feedback aligned with CADMultiTurnScheduler (view task):
        # append rendered multi-view image + compare prompt (conditional DONE, not forced).
        conversation.append({"role": "assistant", "content": resp})
        gen_view_path = exec_result.get('generated_view_path')
        gen_img_url = gen_client.encode_image(gen_view_path) if gen_view_path else None

        if gen_img_url:
            feedback_content = [
                {"type": "text", "text": (
                    "Your generated model views from the previous turn are shown below.\n"
                    "Compare them with the original design request provided at the beginning "
                    "of this conversation.\n\n"
                    "If the model is correct and complete, output <DONE>.\n"
                    "Otherwise, briefly analyze the remaining issues in <thinking></thinking> tags "
                    "and provide corrected code."
                )},
                {"type": "image_url", "image_url": {"url": gen_img_url}},
            ]
        else:
            # Exec ok but render failed / missing — same as training need_render branch.
            feedback_content = (
                "Your code executed successfully, "
                "but the 3D model could not be rendered (possibly empty geometry).\n"
                "Please check that the model produces valid non-empty geometry and fix the code."
            )
        conversation.append({"role": "user", "content": feedback_content})

    return {
        'status': 'success' if last_valid_code else 'fail',
        'task_type': TASK_ITERCAD_DRAW,
        'uid': uid,
        'qid': sample_data.get('qid', ''),
        # Store input GT view path for downstream reproduction/visualization
        'gt_view_path': sample_data.get('gt_view_path', ''),
        'final_cd': final_cd,
        'turns': len(all_turns_data),
        'all_turns_data': all_turns_data,
        'generated_view_paths': generated_view_paths,
        'timestamp': datetime.now().isoformat(),
    }


# ==========================================
# 4. IterCAD-Edit Worker (text instruction only)
# ==========================================

def process_itercad_edit(sample_data, config):
    """IterCAD-Edit: modify code from a text editing instruction."""
    uid = get_uid(sample_data)

    original_code = read_code_file(sample_data.get('input_code_path'))
    if not original_code:
        return {'status': 'skip', 'reason': 'no_original_code', 'uid': uid, 'task_type': TASK_ITERCAD_EDIT}

    instruction = (sample_data.get('instruction_quantitative', '') or
                   sample_data.get('instruction_qualitative', ''))
    if not instruction:
        return {'status': 'skip', 'reason': 'no_instruction', 'uid': uid, 'task_type': TASK_ITERCAD_EDIT}

    gen_client = GeneratorClient(config)
    temp_dir = config['temp_dir']
    os.makedirs(temp_dir, exist_ok=True)
    sample_points = config.get('sample_points', 8192)

    gt_stl_path = sample_data.get('target_stl_path') or sample_data.get('gt_stl_path', '')
    if not gt_stl_path or not os.path.exists(gt_stl_path):
        gt_stl_path = execute_gt_code(sample_data.get('target_code_path'), temp_dir, uid)

    sys_msg = {"role": "system", "content": gen_client.system_prompt}
    initial_user_msg = {
        "role": "user",
        "content": (
            f"### Original CadQuery Code:\n```python\n{original_code}\n```\n\n"
            f"### Editing Instruction:\n{instruction}\n\n"
            "Please modify the code according to the instruction."
        )
    }

    conversation = [sys_msg, initial_user_msg]
    all_turns_data = []
    final_cd = -1.0
    last_valid_code = None

    for turn in range(config['max_turns']):
        resp, error = gen_client.chat(conversation)
        if not resp:
            all_turns_data.append({'turn': turn, 'status': 'api_error', 'error': error})
            continue

        code_str, has_done = extract_code(resp)
        turn_data = {'turn': turn, 'response': resp, 'code': code_str}

        if not code_str and has_done and last_valid_code is not None:
            turn_data['status'] = 'done_by_model'
            all_turns_data.append(turn_data)
            break

        if not code_str:
            turn_data['status'] = 'no_code'
            all_turns_data.append(turn_data)
            conversation.append({"role": "assistant", "content": resp})
            if turn == 0 and has_done and last_valid_code is None:
                user_msg = (
                    "Self-reflection: you output <DONE> before any code was executed. "
                    "In <thinking>, briefly re-check the task, then output one ```python``` block. "
                    "Do not output <DONE> until a later turn after review."
                )
            else:
                user_msg = (
                    "Please output valid python code inside ```python``` block. "
                    "Do NOT output <DONE> before generating code."
                )
            conversation.append({"role": "user", "content": user_msg})
            continue

        exec_result, _ = try_execute_and_eval(
            code_str, None, temp_dir, uid, turn, gt_stl_path, sample_points, need_render=False)
        turn_data.update(exec_result)

        if not exec_result.get('exec_success'):
            turn_data['status'] = 'exec_error'
            all_turns_data.append(turn_data)
            conversation.append({"role": "assistant", "content": resp})
            err = exec_result.get('exec_message', 'Unknown error')
            conversation.append({"role": "user", "content": (
                f"[Turn Feedback] Runtime Error:\n"
                f"```\n{err}\n```\n\n"
                "Feedback Analysis: Precisely identify what failed.\n"
                "Modification Plan: State the targeted local edits required. "
                "Explicitly define what existing code must be preserved.\n\n"
                "Please fix the code."
            )})
            continue

        last_valid_code = code_str
        if 'chamfer_distance' in exec_result:
            final_cd = exec_result['chamfer_distance']

        turn_data['status'] = 'exec_success'
        all_turns_data.append(turn_data)

        if has_done and turn > 0:
            break

        # Text-only feedback aligned with CADMultiTurnScheduler (text / edit task).
        conversation.append({"role": "assistant", "content": resp})
        conversation.append({"role": "user", "content": (
            "Your code from the previous turn executed successfully.\n"
            "Check whether it fully and correctly applies the editing instruction.\n\n"
            "If it is correct and complete, output <DONE>.\n"
            "Otherwise, briefly analyze the remaining issues in <thinking></thinking> tags "
            "and provide corrected code."
        )})

    return {
        'status': 'success' if last_valid_code else 'fail',
        'task_type': TASK_ITERCAD_EDIT,
        'uid': uid,
        'source_file': sample_data.get('source_file', ''),
        'level': sample_data.get('level', ''),
        'transform_type': sample_data.get('transform_type', ''),
        'instruction': instruction,
        'final_cd': final_cd,
        'turns': len(all_turns_data),
        'all_turns_data': all_turns_data,
        'last_valid_code': last_valid_code,
        'timestamp': datetime.now().isoformat(),
    }


# ==========================================
# 5. Unified dispatch
# ==========================================

def process_single_case(sample_data, config):
    task_type = sample_data.get('_task_type', TASK_ITERCAD_DRAW)
    if task_type in (TASK_ITERCAD_EDIT, 'edit'):
        return process_itercad_edit(sample_data, config)
    return process_itercad_draw(sample_data, config)


def get_best_cd_from_result(res):
    """Return final successful CD for one inference run (aligned with compute_metrics)."""
    cd, _ = extract_successful_final_cd(res)
    return cd


def extract_successful_final_cd(res):
    """Success criteria: trajectory ends successfully on the last turn.
    - Last turn exec_success with CD -> use that turn's CD
    - Last turn done_by_model with valid final_cd -> use final_cd
    - Otherwise failed (even if an earlier turn executed successfully)
    - turn_no = len(all_turns_data), full model reply count including DONE turn
    """
    turns = res.get('all_turns_data', [])
    if not turns:
        return None, None

    last = turns[-1]
    if last.get('exec_success') and 'chamfer_distance' in last:
        return last['chamfer_distance'], len(turns)

    if last.get('status') == 'done_by_model':
        final_cd = res.get('final_cd', -1)
        if final_cd is None or final_cd < 0:
            return None, None
        return final_cd, len(turns)

    return None, None


def compute_auc_tr(cds, total, min_cd=1e-6, max_cd=1e-1, num_points=401):
    """AUC-TR: normalized area under CD tolerance recall curve (same as plot_final_cd_recall.py)."""
    if total <= 0 or not cds or num_points < 2 or min_cd <= 0 or max_cd <= 0 or min_cd >= max_cd:
        return 0.0

    x_min = -math.log10(max_cd)
    x_max = -math.log10(min_cd)
    cds_sorted = sorted(cds)
    xs = [x_min + (x_max - x_min) * idx / (num_points - 1) for idx in range(num_points)]
    recalls = []
    for threshold in [10 ** (-x) for x in xs]:
        ok = 0
        for cd in cds_sorted:
            if cd <= threshold:
                ok += 1
            else:
                break
        recalls.append(ok / total)

    area = 0.0
    for idx in range(1, len(xs)):
        dx = xs[idx] - xs[idx - 1]
        area += dx * (recalls[idx] + recalls[idx - 1]) / 2.0
    return area / (x_max - x_min) if x_max > x_min else 0.0


# ==========================================
# 6. Metrics
# ==========================================

def compute_metrics(results, min_cd=1e-6, max_cd=1e-1, num_points=401):
    """
    Metrics:
    - Success: trajectory ends on exec_success or done_by_model
    - AUC-TR: normalized recall@CD tolerance curve area; denominator is all samples
    - avg_turns: mean full model reply count across all samples
    """
    total = len(results)
    if total == 0:
        return {'total_samples': 0}

    successful_cds = []
    all_turn_counts = []
    failed_count = 0

    for res in results:
        turn_count = res.get('turns')
        if turn_count is None:
            turn_count = len(res.get('all_turns_data', []))
        all_turn_counts.append(turn_count)

        final_cd, _ = extract_successful_final_cd(res)
        if final_cd is None:
            failed_count += 1
            continue
        successful_cds.append(final_cd)

    metrics = {
        'total_samples': total,
        'failed_samples': failed_count,
        'failed_rate': failed_count / total if total > 0 else 0.0,
        'successful_samples': len(successful_cds),
        'successful_rate': len(successful_cds) / total if total > 0 else 0.0,
        'auc_tr': float(compute_auc_tr(successful_cds, total, min_cd, max_cd, num_points)),
        'auc_tr_min_cd': min_cd,
        'auc_tr_max_cd': max_cd,
        'auc_tr_num_points': num_points,
        'avg_turns': float(np.mean(all_turn_counts)) if all_turn_counts else 0.0,
    }

    if successful_cds:
        metrics['successful_metrics'] = {
            'mean_cd': float(np.mean(successful_cds)),
            'median_cd': float(np.median(successful_cds)),
            'min_cd': float(np.min(successful_cds)),
            'max_cd': float(np.max(successful_cds)),
        }

    return metrics


def metrics_kwargs_from_config(config):
    return {
        'min_cd': config.get('auc_tr_min_cd', 1e-6),
        'max_cd': config.get('auc_tr_max_cd', 1e-1),
        'num_points': config.get('auc_tr_num_points', 401),
    }


def print_metrics(metrics, label=""):
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}=== Evaluation Results ===")
    print(f"{prefix}Total: {metrics['total_samples']}, "
          f"Successful: {metrics.get('successful_samples', 0)} ({metrics.get('successful_rate', 0):.2%}), "
          f"Failed: {metrics.get('failed_samples', 0)} ({metrics.get('failed_rate', 0):.2%})")

    if 'successful_metrics' in metrics:
        m = metrics['successful_metrics']
        print(f"{prefix}Successful CD*10^3 — Mean: {m['mean_cd']*1000:.6f}, Median: {m['median_cd']*1000:.6f}, "
              f"Min: {m['min_cd']*1000:.6f}, Max: {m['max_cd']*1000:.6f}")

    if 'auc_tr' in metrics:
        print(f"{prefix}AUC-TR: {metrics['auc_tr']:.6f} "
              f"(CD in [{metrics.get('auc_tr_min_cd')}, {metrics.get('auc_tr_max_cd')}])")

    v = metrics.get('avg_turns', -1)
    if v >= 0:
        print(f"{prefix}Avg turns: {v:.2f}")


def _result_status_rank(status):
    return {'success': 2, 'fail': 1, 'skip': 0}.get(status, 0)


def compare_results(a, b):
    """Pick the better result: prefer success, then lower CD."""
    rank_a = _result_status_rank(a.get('status'))
    rank_b = _result_status_rank(b.get('status'))
    if rank_a != rank_b:
        return a if rank_a > rank_b else b

    cd_a = get_best_cd_from_result(a)
    cd_b = get_best_cd_from_result(b)
    if cd_a is not None and cd_b is not None:
        return a if cd_a <= cd_b else b
    if cd_a is not None:
        return a
    if cd_b is not None:
        return b
    return a


def process_single_case_pass_k(sample_data, config):
    """Run pass_k inferences per sample and keep only the best trajectory."""
    pass_k = config.get('pass_k', 1)
    if pass_k <= 1:
        return process_single_case(sample_data, config)

    best = None
    for attempt_idx in range(pass_k):
        attempt_config = dict(config)
        attempt_config['temp_dir'] = os.path.join(config['temp_dir'], f"attempt{attempt_idx}")
        res = process_single_case(sample_data, attempt_config)
        if best is None or compare_results(best, res) is res:
            best = res

    best = dict(best)
    return best


def sanitize_result_record(res):
    """Drop internal pass@k metadata before logging or persisting results."""
    cleaned = dict(res)
    cleaned.pop('pass_k', None)
    cleaned.pop('selected_attempt', None)
    return cleaned


def iter_json_records_from_line(line):
    """Parse one or more JSON objects from a single jsonl line (tolerates glued records)."""
    decoder = json.JSONDecoder()
    idx = 0
    text = line.strip()
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        yield obj
        idx = end


def load_results_jsonl(path):
    """Load eval result records; tolerant to multiple JSON objects on one line."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parsed_any = False
            for rec in iter_json_records_from_line(line):
                records.append(rec)
                parsed_any = True
            if not parsed_any:
                print(f"[WARN] Skipping unparsable result line {line_no} in {path}", flush=True)
    return records


def append_result_record(path, record):
    """Append one sanitized result line with file locking."""
    import fcntl

    payload = json.dumps(sanitize_result_record(record), ensure_ascii=False) + "\n"
    with open(path, 'a', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_processed_uids(result_files, failed_files, task_types, extra_read_paths=None):
    """Load completed uids from result files; extra_read_paths supports legacy task names."""
    processed = {tt: set() for tt in task_types}
    for tt in task_types:
        paths = [result_files[tt], failed_files[tt]]
        if extra_read_paths and tt in extra_read_paths:
            paths.extend(extra_read_paths[tt])
        for path in paths:
            if os.path.exists(path):
                for rec in load_results_jsonl(path):
                    processed[tt].add(rec.get('uid', ''))
    return processed


def filter_remaining_samples(all_samples, processed_uids):
    remaining = []
    for s in all_samples:
        tt = s['_task_type']
        if get_uid(s) not in processed_uids[tt]:
            remaining.append(s)
    return remaining


def load_task_results(result_files, task_type, legacy_read_paths=None):
    """Load all results for a task, merging new/legacy filenames and deduplicating by uid."""
    paths = [result_files[task_type]]
    if legacy_read_paths and task_type in legacy_read_paths:
        paths.extend(legacy_read_paths[task_type])
    merged = []
    seen_uids = set()
    for path in paths:
        if not os.path.exists(path):
            continue
        for rec in load_results_jsonl(path):
            uid = rec.get('uid')
            if uid in seen_uids:
                continue
            seen_uids.add(uid)
            merged.append(rec)
    return merged


def create_process_pool(max_workers):
    """Create process pool; on Python 3.11+ restart worker after each sample to reduce OCC leaks."""
    kwargs = {'max_workers': max_workers}
    if sys.version_info >= (3, 11):
        kwargs['max_tasks_per_child'] = 1
    return ProcessPoolExecutor(**kwargs)


# ==========================================
# 7. Main
# ==========================================

def load_jsonl(path):
    path = os.path.abspath(os.path.expanduser(path))
    base_dir = os.path.dirname(path)
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(normalize_sample_paths(json.loads(line), base_dir))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description='Unified CAD Eval Pipeline (IterCAD-Draw + IterCAD-Edit)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # IterCAD-Draw
  python evalution.py --task_type IterCAD-Draw --gen_model YOUR_MODEL

  # IterCAD-Edit (text instruction only)
  python evalution.py --task_type IterCAD-Edit --gen_model YOUR_MODEL

  # run both tasks
  python evalution.py --task_type both --gen_model YOUR_MODEL \\
      --test_samples_IterCAD-Draw draw.jsonl --test_samples_IterCAD-Edit edit.jsonl
        """
    )
    parser.add_argument('--task_type', type=str, default=TASK_ITERCAD_DRAW,
                        choices=[TASK_ITERCAD_DRAW, TASK_ITERCAD_EDIT, 'edit', 'both'],
                        help="'edit' is a legacy alias for IterCAD-Edit")
    parser.add_argument('--test_samples_IterCAD-Draw', dest='test_samples_itercad_draw',
                        type=str, default=DEFAULT_ITERCAD_DRAW_JSONL)
    parser.add_argument('--test_samples_IterCAD-Edit', '--test_samples_edit',
                        dest='test_samples_itercad_edit',
                        type=str, default=DEFAULT_ITERCAD_EDIT_JSONL,
                        help='IterCAD-Edit benchmark jsonl (--test_samples_edit is legacy alias)')
    parser.add_argument('--generator_api', type=str, default=None)
    parser.add_argument('--generator_api_key', type=str, default=None)
    parser.add_argument('--gen_model', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Parallel worker count (default 4; CadQuery/OCC may crash above 8)')
    parser.add_argument('--queue_size', type=int, default=None,
                        help='Pending task queue size (default 8)')
    parser.add_argument('--max_turns', type=int, default=5)
    parser.add_argument('--pass_k', type=int, default=None,
                        help='Internal: run multiple inferences per sample and keep the best (not written to logs/results)')
    parser.add_argument('--no_pool_recover', action='store_true',
                        help='Do not auto-restart process pool after crash (default: auto-restart and resume)')
    parser.add_argument('--auc_tr_min_cd', type=float, default=None)
    parser.add_argument('--auc_tr_max_cd', type=float, default=None)
    parser.add_argument('--auc_tr_num_points', type=int, default=None)
    parser.add_argument('--temp_dir', type=str, default=None)
    parser.add_argument('--extra_body', type=str, default=None,
                        help='vLLM extra_body JSON, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    parser.add_argument(
        '--run_id',
        type=str,
        default=None,
        help='Result file suffix (fixed string). Use the same value when resuming; '
             'otherwise a new timestamped file is created. '
             'Defaults to YYYYMMDD_HHMMSS when omitted.',
    )
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    for key, val in [
        ('generator_api', args.generator_api),
        ('generator_api_key', args.generator_api_key),
        ('gen_model_name', args.gen_model),
        ('output_dir', args.output_dir),
        ('max_workers', args.max_workers),
        ('queue_size', args.queue_size),
        ('max_turns', args.max_turns),
        ('pass_k', args.pass_k),
        ('auc_tr_min_cd', args.auc_tr_min_cd),
        ('auc_tr_max_cd', args.auc_tr_max_cd),
        ('auc_tr_num_points', args.auc_tr_num_points),
        ('temp_dir', args.temp_dir),
    ]:
        if val is not None:
            config[key] = val

    if not config['gen_model_name']:
        parser.error("Generator model is required. Set GEN_MODEL or pass --gen_model.")

    if args.extra_body:
        try:
            config['extra_body'] = json.loads(args.extra_body)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid --extra_body JSON: {e}")
            return

    save_name = config['gen_model_name'].replace('/', '_')
    if args.output_dir is None:
        config['output_dir'] = f"results_{save_name}_unified"
    os.makedirs(config['output_dir'], exist_ok=True)

    # ---- Load samples ----
    task_type = TASK_ITERCAD_EDIT if args.task_type == 'edit' else args.task_type
    run_tasks = [TASK_ITERCAD_DRAW, TASK_ITERCAD_EDIT] if task_type == 'both' else [task_type]
    all_samples = []

    if TASK_ITERCAD_DRAW in run_tasks and args.test_samples_itercad_draw:
        draw_samples = load_jsonl(args.test_samples_itercad_draw)
        for s in draw_samples:
            s['_task_type'] = TASK_ITERCAD_DRAW
        all_samples.extend(draw_samples)
        print(f"Loaded {len(draw_samples)} IterCAD-Draw samples from {args.test_samples_itercad_draw}")

    if TASK_ITERCAD_EDIT in run_tasks and args.test_samples_itercad_edit:
        edits = load_jsonl(args.test_samples_itercad_edit)
        for s in edits:
            s['_task_type'] = TASK_ITERCAD_EDIT
        all_samples.extend(edits)
        print(f"Loaded {len(edits)} IterCAD-Edit samples from {args.test_samples_itercad_edit}")

    if not all_samples:
        print("No samples loaded. Exiting.")
        return

    # ---- Result files ----
    actual_tasks = list(set(s['_task_type'] for s in all_samples))
    # Default to timestamp to avoid overwrite; --run_id fixes filenames for resume
    run_suffix = args.run_id if args.run_id else datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.run_id:
        print(f"Run suffix (resume): {run_suffix}")
    else:
        print(f"Run suffix (new files): {run_suffix}")

    result_files, failed_files, metrics_files = {}, {}, {}
    legacy_read_paths = {}
    for tt in actual_tasks:
        result_files[tt] = os.path.join(config['output_dir'], f'results_{save_name}_{tt}_{run_suffix}.jsonl')
        failed_files[tt] = os.path.join(config['output_dir'], f'failed_{save_name}_{tt}_{run_suffix}.jsonl')
        metrics_files[tt] = os.path.join(config['output_dir'], f'metrics_{save_name}_{tt}_{run_suffix}.json')
        aliases = TASK_LEGACY_ALIASES.get(tt, [])
        if aliases:
            legacy_read_paths[tt] = []
            for alias in aliases:
                legacy_read_paths[tt].append(
                    os.path.join(config['output_dir'], f'results_{save_name}_{alias}_{run_suffix}.jsonl')
                )
                legacy_read_paths[tt].append(
                    os.path.join(config['output_dir'], f'failed_{save_name}_{alias}_{run_suffix}.jsonl')
                )

    # ---- Resume from the same result files (same run_suffix) ----
    processed_uids = load_processed_uids(
        result_files, failed_files, actual_tasks, extra_read_paths=legacy_read_paths
    )
    remaining = filter_remaining_samples(all_samples, processed_uids)

    print(f"\nTotal: {len(all_samples)}, Remaining: {len(remaining)}")

    if not remaining:
        print("All samples processed. Computing metrics from existing files...")
        for tt in result_files:
            existing = load_task_results(result_files, tt, legacy_read_paths)
            if not existing:
                print(f"\n[{tt}] No results to compute metrics.")
                continue
            metrics = compute_metrics(existing, **metrics_kwargs_from_config(config))
            with open(metrics_files[tt], 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print_metrics(metrics, label=tt)
            print(f"  Results: {result_files[tt]}")
            print(f"  Failed:  {failed_files[tt]}")
            print(f"  Metrics: {metrics_files[tt]}")
        return

    # ---- Parallel execution (auto-restart pool on crash) ----
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    pool_recover = not args.no_pool_recover

    print(f"Starting evaluation (workers={config['max_workers']}, "
          f"queue_size={config['queue_size']}, max_turns={config['max_turns']}, "
          f"pool_recover={'on' if pool_recover else 'off'})...\n")

    def handle_result(res):
        tt = res.get('task_type', TASK_ITERCAD_DRAW)
        uid = res.get('uid', 'unknown')
        status = res.get('status')

        target_file = result_files[tt] if status in ('success', 'fail') else failed_files[tt]
        if status == 'skip':
            print(f"[SKIP][{tt}] [{uid}] {res.get('reason')}", flush=True)
            return

        label = 'OK' if status == 'success' else 'FAIL'
        cd_str = f" CD={res.get('final_cd', -1):.6f}" if res.get('final_cd', -1) >= 0 else ""
        print(f"[{label}][{tt}] [{uid}] Turns={res.get('turns', 0)}{cd_str}", flush=True)

        with lock:
            append_result_record(target_file, res)

    def _drain_future(fut):
        try:
            handle_result(fut.result())
        except BrokenProcessPool:
            raise
        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()

    def _run_pool_batch(work_items):
        with create_process_pool(config['max_workers']) as pool:
            pending = {}
            for sample in work_items:
                while len(pending) >= config['queue_size']:
                    done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                    for fut in done:
                        _drain_future(fut)
                    pending = {f: pending[f] for f in pending if f not in done}

                pending[pool.submit(process_single_case_pass_k, sample, config)] = sample

            while pending:
                done, _ = wait(list(pending.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    _drain_future(fut)
                pending = {f: pending[f] for f in pending if f not in done}

    pool_crashes = 0
    while remaining:
        try:
            _run_pool_batch(remaining)
            break
        except BrokenProcessPool:
            pool_crashes += 1
            processed_uids = load_processed_uids(
                result_files, failed_files, actual_tasks, extra_read_paths=legacy_read_paths
            )
            remaining = filter_remaining_samples(all_samples, processed_uids)
            if not remaining:
                print("[INFO] All remaining samples were written before pool crash; computing metrics.", flush=True)
                break
            if not pool_recover:
                print(
                    "[FATAL] Process pool is broken; run ended early. Completed samples are already appended to jsonl. "
                    "Reduce --max_workers and resume with the same --run_id.",
                    flush=True,
                )
                return
            print(
                f"[WARN] Process pool crashed (#{pool_crashes}), often due to CadQuery/OCC segfault or OOM. "
                f"Auto-restarting in 5s, {len(remaining)} samples remaining...",
                flush=True,
            )
            time.sleep(5)

    if pool_crashes:
        print(f"[INFO] Auto-recovered process pool {pool_crashes} time(s) in this run.", flush=True)

    # ---- Compute metrics ----
    print("\n\nComputing metrics...")
    for tt in result_files:
        all_for_metrics = load_task_results(result_files, tt, legacy_read_paths)
        if not all_for_metrics:
            print(f"\n[{tt}] No results to compute metrics.")
            continue

        metrics = compute_metrics(all_for_metrics, **metrics_kwargs_from_config(config))
        with open(metrics_files[tt], 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print_metrics(metrics, label=tt)
        print(f"  Results: {result_files[tt]}")
        print(f"  Failed:  {failed_files[tt]}")
        print(f"  Metrics: {metrics_files[tt]}")


if __name__ == "__main__":
    main()
