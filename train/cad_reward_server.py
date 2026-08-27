#!/usr/bin/env python3
"""
CAD Reward API Server

负责: 代码执行 / 三视图渲染 / Chamfer Distance 计算
启动: python cad_reward_server.py --port 8765

GRPO 训练时单独部署此服务，reward function / scheduler 通过 HTTP 调用。
"""

import os
import sys
import base64
import tempfile
import argparse
import traceback
import json
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

app = FastAPI(title="CAD Reward API")
_max_workers = int(os.environ.get("CAD_SERVER_WORKERS", "16"))
_executor = ThreadPoolExecutor(max_workers=_max_workers)
_eval_semaphore = threading.BoundedSemaphore(_max_workers)

# code_executor 模块（在 startup 时 import）
_ce = None


@app.on_event("startup")
def _load_modules():
    global _ce
    # 默认隔离执行：code_executor/CadQuery 只在子进程中 import。
    # 这样 native segfault 只会杀掉单个 worker，不会杀掉 HTTP 服务。
    if os.environ.get("CAD_ISOLATE_EVAL", "1") != "1":
        import code_executor as ce
        _ce = ce


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    code: str
    gt_stl_path: Optional[str] = None
    sample_points: int = 8192
    render: bool = False           # 是否返回渲染三视图 base64


class EvalResponse(BaseModel):
    exec_ok: bool
    exec_msg: str
    cd: Optional[float] = None
    render_b64: Optional[str] = None   # PNG base64，仅 render=True 时返回


# ---------------------------------------------------------------------------
# 核心逻辑（同步，在线程池中运行）
# ---------------------------------------------------------------------------

def _evaluate_with_executor(req: EvalRequest, ce) -> EvalResponse:
    # 1. 执行代码
    exec_ok, exec_msg, cq_obj = ce.execute_code(req.code)
    if not exec_ok:
        return EvalResponse(exec_ok=False, exec_msg=exec_msg)

    # 2. 计算 Chamfer Distance
    cd = None
    if req.gt_stl_path and os.path.exists(req.gt_stl_path):
        try:
            cd = ce.compute_cd(cq_obj, req.gt_stl_path, req.sample_points)
        except Exception:
            pass

    # 3. 可选渲染三视图
    render_b64 = None
    if req.render:
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            if ce.render_views(cq_obj, tmp_path):
                with open(tmp_path, "rb") as f:
                    render_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            traceback.print_exc()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return EvalResponse(exec_ok=True, exec_msg="Success", cd=cd, render_b64=render_b64)


def _write_worker_result(result_path: str, status: str, payload: dict):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"status": status, "payload": payload}, f, ensure_ascii=False)


def _evaluate_worker(payload: dict, eval_dir: str, result_path: str):
    """在独立进程中执行 CAD 评估，隔离 CadQuery/OpenCASCADE 的 native 崩溃。"""
    try:
        sys.path.insert(0, eval_dir)
        import code_executor as ce

        req = EvalRequest(**payload)
        result = _evaluate_with_executor(req, ce)
        _write_worker_result(result_path, "ok", result.dict())
    except BaseException as exc:
        traceback.print_exc()
        _write_worker_result(
            result_path,
            "error",
            {
                "exec_ok": False,
                "exec_msg": f"{type(exc).__name__}: {exc}",
                "cd": None,
                "render_b64": None,
            },
        )


def _evaluate_isolated(req: EvalRequest) -> EvalResponse:
    timeout = float(os.environ.get("CAD_EVAL_TIMEOUT", "80"))
    eval_dir = os.environ.get("CAD_EVAL_DIR", os.path.dirname(os.path.abspath(__file__)))
    mp_context = os.environ.get("CAD_MP_CONTEXT", "spawn")
    ctx = mp.get_context(mp_context)
    payload = req.dict()
    fd, result_path = tempfile.mkstemp(prefix="cad_eval_result_", suffix=".json")
    os.close(fd)

    try:
        proc = ctx.Process(target=_evaluate_worker, args=(payload, eval_dir, result_path))
        proc.start()
        proc.join(timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            return EvalResponse(
                exec_ok=False,
                exec_msg=f"CAD evaluation timed out after {timeout:.1f}s",
            )

        if proc.exitcode != 0:
            return EvalResponse(
                exec_ok=False,
                exec_msg=f"CAD evaluation worker crashed with exit code {proc.exitcode}",
            )

        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
            status = result.get("status")
            payload = result.get("payload", {})
        except Exception:
            return EvalResponse(
                exec_ok=False,
                exec_msg="CAD evaluation worker exited without a result",
            )

        if status != "ok":
            return EvalResponse(**payload)
        return EvalResponse(**payload)
    finally:
        if os.path.exists(result_path):
            os.remove(result_path)


def _evaluate_sync(req: EvalRequest) -> EvalResponse:
    with _eval_semaphore:
        if os.environ.get("CAD_ISOLATE_EVAL", "1") == "1":
            return _evaluate_isolated(req)
        return _evaluate_with_executor(req, _ce)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/evaluate", response_model=EvalResponse)
async def evaluate(req: EvalRequest):
    return await run_in_threadpool(_evaluate_sync, req)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default_eval_dir = os.path.join(_repo_root, "eval")

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--eval_dir",
        type=str,
        default=_default_eval_dir,
        help="包含 code_executor.py / gen_view.py 的目录（默认: <repo>/eval）",
    )
    args = parser.parse_args()

    if not os.path.isfile(os.path.join(args.eval_dir, "code_executor.py")):
        raise SystemExit(
            f"code_executor.py not found under --eval_dir: {args.eval_dir}"
        )

    sys.path.insert(0, args.eval_dir)
    os.environ.setdefault("CAD_EVAL_DIR", args.eval_dir)
    uvicorn.run(app, host=args.host, port=args.port)
