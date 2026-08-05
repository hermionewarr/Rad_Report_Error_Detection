#!/usr/bin/env python3
"""Run the same local LLM eval across multiple models and build a comparison table.

Designed for privacy: only talks to loopback unless you pass --allow_remote on
the underlying eval_llm_baseline.py subprocess (not exposed here by default).

Typical usage (from project root, Rad_Report_Error_Detection-LLM conda env active):

  python scripts/run_llm_baselines.py \\
    --eval_csv data/eval_datasets/hard_negative_eval_v3.csv \\
    --mode lat \\
    --suite_name hard-v3-lat \\
    --manage_vllm \\
    --bert_predictions trained_models/runs/.../predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional

BERT_MODEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BERT_MODEL_ROOT  # flat project root
SRC_DIR = BERT_MODEL_ROOT / "src"
RUNS_DIR = BERT_MODEL_ROOT / "trained_models" / "runs" / "llm_baseline"
DEFAULT_MODELS_JSON = BERT_MODEL_ROOT / "config" / "llm_models_default.json"

EVAL_SCRIPT = SRC_DIR / "eval_llm_baseline.py"
METRICS_SCRIPT = SRC_DIR / "llm_eval_metrics.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.llm_eval_prompts import USER_PROMPT_TEMPLATES  # noqa: E402
from src.llm_eval_metrics import (  # noqa: E402
    incidence_modes_for_eval_mode,
    incidence_report_filename,
    summarize_predictions_csv,
)


@contextmanager
def _suite_run_lock(suite_root: Path) -> Iterator[Path]:
    """Prevent concurrent runs from sharing one suite directory."""
    suite_root.mkdir(parents=True, exist_ok=True)
    lock_path = suite_root / ".run.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        lock_file.write(f"pid={os.getpid()}\n")
        lock_file.flush()
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another run_llm_baselines process holds {lock_path}. "
                "Stop the other sweep before starting a new one."
            ) from exc
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_models_json(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"models JSON must be a non-empty list: {path}")
    for i, item in enumerate(data):
        if "slug" not in item or "model" not in item:
            raise ValueError(f"models[{i}] needs 'slug' and 'model': {item}")
    return data


def _wait_for_vllm(base_url: str, timeout_s: float = 1800.0) -> None:
    url = base_url.rstrip("/") + "/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"vLLM ready at {base_url}")
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not become ready within {timeout_s}s ({url})")


def _apply_vllm_runtime_env(env: Dict[str, str], cuda_visible_devices: Optional[str]) -> None:
    """Env vars for the vLLM server subprocess."""
    # scipy/transformers wheels in the conda env need a newer libstdc++ (CXXABI_1.3.15)
    # than Ubuntu's /lib/x86_64-linux-gnu/libstdc++.so.6. Prefer conda's copy.
    conda_prefix = env.get("CONDA_PREFIX") or os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        existing = env.get("LD_LIBRARY_PATH", "")
        if conda_lib not in existing.split(":"):
            env["LD_LIBRARY_PATH"] = (
                f"{conda_lib}:{existing}" if existing else conda_lib
            )
            print(f"Prepended LD_LIBRARY_PATH with {conda_lib}")

    # Stable GPU numbering when multiple NVIDIA cards are present.
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        print(f"Using CUDA_VISIBLE_DEVICES={cuda_visible_devices}")

    # FlashInfer's sampler JIT-compiles CUDA kernels and needs nvcc. Many dev boxes
    # only have the driver + PyTorch CUDA wheels, not a full toolkit at /usr/local/cuda.
    # PyTorch-native sampling is fine for our low-throughput JSON classification eval.
    if "VLLM_USE_FLASHINFER_SAMPLER" not in env:
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        print("Set VLLM_USE_FLASHINFER_SAMPLER=0 (no nvcc required for sampling)")

    # If nvcc exists (conda cuda-nvcc or system toolkit), point FlashInfer at it.
    if "CUDA_HOME" not in env:
        for candidate in (
            os.environ.get("CONDA_PREFIX"),
            "/usr/local/cuda",
            "/usr/local/cuda-12",
            "/usr/local/cuda-13",
        ):
            if candidate and Path(candidate, "bin", "nvcc").is_file():
                env["CUDA_HOME"] = candidate
                print(f"Set CUDA_HOME={candidate}")
                break


def _is_gguf_model(model_id: str, spec: Dict) -> bool:
    if spec.get("load_format") == "gguf" or spec.get("gguf_file"):
        return True
    lower = model_id.lower()
    return lower.endswith(".gguf") or "-gguf" in lower


def _warn_quantization_mismatch(model_id: str, quantization: Optional[str], spec: Optional[Dict] = None) -> None:
    if not quantization:
        return
    if _is_gguf_model(model_id, spec or {}):
        return
    lower = model_id.lower()
    if quantization == "awq" and "awq" not in lower and "gptq" not in lower:
        print(
            f"WARNING: quantization={quantization!r} but model id {model_id!r} looks like "
            "full-precision weights. Use the matching *-AWQ (or GPTQ) Hugging Face repo, "
            "e.g. Qwen/Qwen2.5-72B-Instruct-AWQ."
        )


def _model_enabled(spec: Dict) -> bool:
    if spec.get("enabled") is False or spec.get("disabled") is True:
        return False
    return True


def _suite_model_paths(model_dir: Path) -> Dict[str, Path]:
    """Standard per-model layout: results/, plots/, logs/ under model_dir."""
    results_dir = model_dir / "results"
    plots_dir = model_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "plots_dir": plots_dir,
        "predictions_csv": results_dir / "predictions.csv",
        "report_txt": results_dir / "eval_summary.txt",
        "confusion_plot": plots_dir / "confusion_matrix.png",
    }


def _resolve_predictions_csv(model_dir: Path) -> Path:
    """Prefer results/predictions.csv; accept legacy flat layout."""
    paths = _suite_model_paths(model_dir)
    preferred = paths["predictions_csv"]
    legacy = model_dir / "predictions.csv"
    if preferred.is_file():
        return preferred
    if legacy.is_file():
        return legacy
    return preferred


def _resolve_repo_path(path: str) -> str:
    """Resolve project-relative paths (e.g. ...) for vLLM."""
    candidate = Path(path)
    if not candidate.is_absolute():
        rooted = (REPO_ROOT / candidate).resolve()
        if rooted.exists():
            return str(rooted)
    return path


def _start_vllm_server(
    model: str,
    host: str,
    port: int,
    quantization: Optional[str],
    gpu_memory_utilization: float,
    max_model_len: int,
    env: Dict[str, str],
    spec: Optional[Dict] = None,
) -> subprocess.Popen:
    spec = spec or {}
    _warn_quantization_mismatch(model, quantization, spec)
    model = _resolve_repo_path(model)
    hf_config_path = spec.get("hf_config_path")
    if hf_config_path:
        hf_config_path = _resolve_repo_path(str(hf_config_path))
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
    ]
    if quantization and not _is_gguf_model(model, spec):
        cmd.extend(["--quantization", quantization])
    if spec.get("tokenizer"):
        cmd.extend(["--tokenizer", str(spec["tokenizer"])])
    if hf_config_path:
        cmd.extend(["--hf-config-path", hf_config_path])
    if spec.get("dtype"):
        cmd.extend(["--dtype", str(spec["dtype"])])
    if spec.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if spec.get("enforce_eager"):
        cmd.append("--enforce-eager")
    if spec.get("tensor_parallel_size"):
        cmd.extend(["--tensor-parallel-size", str(int(spec["tensor_parallel_size"]))])
    if spec.get("language_model_only"):
        cmd.append("--language-model-only")
    limit_mm = spec.get("limit_mm_per_prompt")
    if limit_mm is not None:
        cmd.extend(["--limit-mm-per-prompt", json.dumps(limit_mm)])
    hf_overrides = spec.get("hf_overrides")
    if hf_overrides:
        cmd.extend(["--hf-overrides", json.dumps(hf_overrides)])
    chat_template_kwargs = spec.get("chat_template_kwargs")
    if chat_template_kwargs:
        cmd.extend(
            ["--default-chat-template-kwargs", json.dumps(chat_template_kwargs)]
        )
    reasoning_parser = spec.get("reasoning_parser")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", str(reasoning_parser)])
    vllm_extra_args = spec.get("vllm_extra_args")
    if vllm_extra_args:
        cmd.extend([str(a) for a in vllm_extra_args])
    print("Starting vLLM:", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    return proc


def _stop_process(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    print("Stopping vLLM server...")
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def _run_subprocess(cmd: List[str], env: Dict[str, str]) -> int:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    return int(result.returncode)


def _fmt_pct(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.1f}%"


def _write_comparison_table(
    rows: List[Dict[str, object]],
    out_csv: Path,
    out_md: Path,
    suite_meta: Dict[str, object],
) -> None:
    fieldnames = [
        "model_slug",
        "model_id",
        "n_total",
        "accuracy_pct",
        "detection_rate_pct",
        "false_alarm_rate_pct",
        "parse_ok_pct",
        "lat_detection_inc1_pct",
        "lat_detection_inc_ge2_pct",
        "predictions_csv",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_lines = [
        "# LLM baseline comparison",
        "",
        f"- **Suite**: `{suite_meta.get('suite_name', '')}`",
        f"- **Eval CSV**: `{suite_meta.get('eval_csv', '')}`",
        f"- **Mode**: `{suite_meta.get('mode', '')}`",
        f"- **Prompt**: `{suite_meta.get('prompt', '1')}`",
        f"- **Generated**: {suite_meta.get('generated', '')}",
        "",
        "| Model | N | Accuracy | Detection | False alarm | Parse OK | Lat det. (inc=1) | Lat det. (inc≥2) |",
        "|-------|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {model_slug} | {n_total} | {accuracy_pct} | {detection_rate_pct} | "
            "{false_alarm_rate_pct} | {parse_ok_pct} | {lat_detection_inc1_pct} | "
            "{lat_detection_inc_ge2_pct} |".format(
                model_slug=row.get("model_slug", ""),
                n_total=row.get("n_total", ""),
                accuracy_pct=_fmt_pct(float(row.get("accuracy_pct", float("nan")))),
                detection_rate_pct=_fmt_pct(float(row.get("detection_rate_pct", float("nan")))),
                false_alarm_rate_pct=_fmt_pct(float(row.get("false_alarm_rate_pct", float("nan")))),
                parse_ok_pct=_fmt_pct(float(row.get("parse_ok_pct", float("nan")))),
                lat_detection_inc1_pct=_fmt_pct(
                    float(row.get("lat_detection_inc1_pct", float("nan")))
                ),
                lat_detection_inc_ge2_pct=_fmt_pct(
                    float(row.get("lat_detection_inc_ge2_pct", float("nan")))
                ),
            )
        )
    md_lines.append("")
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


def _write_comparison_by_error_type(
    model_summaries: List[Dict[str, object]],
    out_csv: Path,
) -> None:
    error_types = sorted(
        {
            error_type
            for summary in model_summaries
            for error_type in (summary.get("by_error_type") or {})
        }
    )
    fieldnames = [
        "model_slug",
        "model_id",
        "error_type",
        "n",
        "detection_or_false_alarm",
        "rate_pct",
        "accuracy_pct",
    ]
    rows: List[Dict[str, object]] = []
    for summary in model_summaries:
        by_type = summary.get("by_error_type") or {}
        for error_type in error_types:
            stats = by_type.get(error_type)
            if not stats:
                continue
            if error_type == "consistent":
                metric_val = int(stats.get("false_alarm", 0))
                rate = stats.get("false_alarm_rate_pct", float("nan"))
            else:
                metric_val = int(stats.get("detected", 0))
                rate = stats.get("detection_rate_pct", float("nan"))
            rows.append(
                {
                    "model_slug": summary.get("model_slug", ""),
                    "model_id": summary.get("model_id", ""),
                    "error_type": error_type,
                    "n": int(stats.get("n", 0)),
                    "detection_or_false_alarm": metric_val,
                    "rate_pct": rate,
                    "accuracy_pct": stats.get("accuracy_pct", float("nan")),
                }
            )
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run multi-model local LLM baseline suite.")
    p.add_argument("--eval_csv", type=str, required=True, help="Eval dataset CSV path.")
    p.add_argument(
        "--mode",
        type=str,
        default="hard",
        choices=["all", "lat", "site", "severity", "hard", "cross_pair_full"],
        help=(
            "Row filter passed to eval_llm_baseline.py: hard = all error types in CSV "
            "(paper default); lat/site/severity = consistent + that swap type only; "
            "cross_pair_full = consistent + cross_pair_mismatch only; all = no filter."
        ),
    )
    p.add_argument(
        "--suite_name",
        type=str,
        required=True,
        help="Short label for this benchmark (used in run folder name).",
    )
    p.add_argument(
        "--models_json",
        type=str,
        default=str(DEFAULT_MODELS_JSON),
        help="JSON list of {slug, model, quantization?} objects.",
    )
    p.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    p.add_argument("--vllm_host", type=str, default="127.0.0.1")
    p.add_argument("--vllm_port", type=int, default=8000)
    p.add_argument(
        "--manage_vllm",
        action="store_true",
        help="Start/stop vLLM per model (default: expect server already running).",
    )
    p.add_argument(
        "--no_manage_vllm",
        action="store_true",
        help="Do not start vLLM; only call eval API (alias for omitting --manage_vllm).",
    )
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        metavar="ID",
        help=(
            "Physical GPU index for vLLM only (sets CUDA_VISIBLE_DEVICES). "
            "Example: --gpu 1 uses the second GPU and leaves GPU 0 free."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke test: first N rows only after mode/error_type filtering.",
    )
    p.add_argument(
        "--error_types",
        type=str,
        default=None,
        help=(
            "Comma-separated error_type filter for eval_llm_baseline.py. "
            "Example: NF_v_Dis_swap for the 512 human-labeled v6 NF vs disease rows."
        ),
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "Optional predictions output path passed to eval_llm_baseline.py "
            "(default: results/predictions.csv under the model output dir)."
        ),
    )
    p.add_argument(
        "--model_output_dir",
        type=str,
        default=None,
        help=(
            "Reuse an existing per-model run directory (e.g. .../models/qwen3-32b-gguf) "
            "instead of creating a new suite subfolder."
        ),
    )
    p.add_argument(
        "--skip-on-failure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with remaining models if one fails (default: on).",
    )
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip model if predictions.csv already exists in suite folder.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Pass --resume to eval_llm_baseline.py (continue from "
            "predictions.checkpoint.csv in the model output dir)."
        ),
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=None,
        help="Pass --checkpoint_every to eval_llm_baseline.py (default: eval script default).",
    )
    p.add_argument(
        "--bert_predictions",
        type=str,
        default=None,
        help="Optional BERT predictions.csv to include as a comparison row.",
    )
    p.add_argument(
        "--bert_label",
        type=str,
        default="bert-lat-v1",
        help="Display name for --bert_predictions row in comparison table.",
    )
    p.add_argument(
        "--report_style",
        type=str,
        default="summary",
        choices=["summary", "incidence"],
        help=(
            "summary = overall + per error_type plus lat/site/severity incidence reports "
            "for hard/all runs (default). incidence = incidence reports only."
        ),
    )
    p.add_argument(
        "--no_incidence_reports",
        action="store_true",
        help="Skip incidence_detection_report*.txt and incidence plots in metrics step.",
    )
    p.add_argument(
        "--use_flashinfer_sampler",
        action="store_true",
        help=(
            "Enable FlashInfer sampling (faster; requires nvcc / CUDA toolkit). "
            "Default is off so eval runs without a system CUDA install."
        ),
    )
    p.add_argument(
        "--prompt",
        type=str,
        choices=sorted(USER_PROMPT_TEMPLATES),
        default="1",
        help=(
            "Built-in user prompt passed to eval_llm_baseline.py: "
            "1=question-style (default), 2=classify-style."
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    manage_vllm = args.manage_vllm and not args.no_manage_vllm
    skip_on_failure = args.skip_on_failure

    if not EVAL_SCRIPT.is_file() or not METRICS_SCRIPT.is_file():
        raise FileNotFoundError("Expected eval scripts under src/")

    models = _load_models_json(Path(args.models_json))
    date_prefix = datetime.now().strftime("%Y%m%d")
    suite_root = RUNS_DIR / f"{date_prefix}_llm-suite_{args.suite_name}"
    models_dir = suite_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"User prompt variant: {args.prompt}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    vllm_env = env.copy()
    if args.use_flashinfer_sampler:
        vllm_env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
    _apply_vllm_runtime_env(vllm_env, args.gpu)

    comparison_rows: List[Dict[str, object]] = []
    manifest_models: List[Dict[str, object]] = []
    skipped_models: List[Dict[str, object]] = []
    vllm_proc: Optional[subprocess.Popen] = None

    def _record_skip(slug: str, model_id: str, reason: str, quant: Optional[str]) -> None:
        skipped_models.append({"slug": slug, "model_id": model_id, "reason": reason})
        manifest_models.append(
            {
                "slug": slug,
                "model_id": model_id,
                "quantization": quant,
                "status": "skipped",
                "reason": reason,
            }
        )
        print(f"SKIP {slug}: {reason}")

    try:
        with _suite_run_lock(suite_root):
            for spec in models:
                slug = spec["slug"]
                # Resolve repo-relative paths so vLLM serve name matches eval API requests.
                model_id = _resolve_repo_path(str(spec["model"]))
                quant = spec.get("quantization")

                if not _model_enabled(spec):
                    _record_skip(slug, model_id, "disabled in models JSON", quant)
                    continue

                model_dir = (
                    Path(args.model_output_dir).resolve()
                    if args.model_output_dir
                    else models_dir / slug
                )
                model_dir.mkdir(parents=True, exist_ok=True)
                suite_paths = _suite_model_paths(model_dir)
                predictions_csv = (
                    Path(args.output_csv).resolve()
                    if args.output_csv
                    else suite_paths["predictions_csv"]
                )
                report_txt = suite_paths["results_dir"] / (
                    "incidence_detection_report.txt"
                    if args.report_style == "incidence"
                    else "eval_summary.txt"
                )
                model_gpu = spec.get("cuda_visible_devices", args.gpu)
                model_vllm_env = env.copy()
                if args.use_flashinfer_sampler:
                    model_vllm_env["VLLM_USE_FLASHINFER_SAMPLER"] = "1"
                _apply_vllm_runtime_env(model_vllm_env, model_gpu)

                ran_ok = False
                try:
                    skip_target = (
                        predictions_csv
                        if args.output_csv
                        else _resolve_predictions_csv(model_dir)
                    )
                    if args.skip_existing and skip_target.is_file():
                        print(f"Skipping {slug}: {skip_target} exists")
                        predictions_csv = skip_target
                        ran_ok = True
                    else:
                        if manage_vllm:
                            _stop_process(vllm_proc)
                            model_max_len = int(spec.get("max_model_len", args.max_model_len))
                            vllm_proc = _start_vllm_server(
                                model=model_id,
                                host=args.vllm_host,
                                port=args.vllm_port,
                                quantization=quant,
                                gpu_memory_utilization=args.gpu_memory_utilization,
                                max_model_len=model_max_len,
                                env=model_vllm_env,
                                spec=spec,
                            )
                            if vllm_proc.poll() is not None:
                                raise RuntimeError(
                                    f"vLLM exited immediately with code {vllm_proc.returncode}"
                                )
                            _wait_for_vllm(f"http://{args.vllm_host}:{args.vllm_port}/v1")

                        eval_cmd = [
                            sys.executable,
                            str(EVAL_SCRIPT),
                            "--eval_csv",
                            args.eval_csv,
                            "--mode",
                            args.mode,
                            "--prompt",
                            args.prompt,
                            "--base_url",
                            args.base_url,
                            "--model",
                            model_id,
                            "--output_run_dir",
                            str(model_dir),
                        ]
                        if args.output_csv:
                            eval_cmd.extend(["--output_csv", str(predictions_csv)])
                        if args.error_types:
                            eval_cmd.extend(["--error_types", args.error_types])
                        if args.limit:
                            eval_cmd.extend(["--limit", str(args.limit)])
                        max_tokens = spec.get("max_tokens")
                        if max_tokens is not None:
                            eval_cmd.extend(["--max_tokens", str(int(max_tokens))])
                        chat_kwargs = spec.get("chat_template_kwargs")
                        if chat_kwargs:
                            eval_cmd.extend(
                                ["--chat_template_kwargs", json.dumps(chat_kwargs)]
                            )
                        if args.resume:
                            eval_cmd.append("--resume")
                        if args.checkpoint_every is not None:
                            eval_cmd.extend(["--checkpoint_every", str(args.checkpoint_every)])
                        if _run_subprocess(eval_cmd, env) != 0:
                            raise RuntimeError("eval_llm_baseline.py failed")

                        predictions_csv = (
                            Path(args.output_csv).resolve()
                            if args.output_csv
                            else _resolve_predictions_csv(model_dir)
                        )
                        metrics_cmd = [
                            sys.executable,
                            str(METRICS_SCRIPT),
                            "--predictions_csv",
                            str(predictions_csv),
                            "--eval_csv",
                            args.eval_csv,
                            "--mode",
                            args.mode,
                            "--prompt",
                            args.prompt,
                            "--report_style",
                            args.report_style,
                            "--output_run_dir",
                            str(model_dir),
                            "--log_file",
                            "metrics.log",
                            "--command_stem",
                            "command_metrics",
                            "--run_name",
                            f"{date_prefix}_llm-suite_{args.suite_name}/{slug}",
                            "--model_name",
                            model_id,
                        ]
                        if args.output_csv:
                            metrics_cmd.extend(
                                [
                                    "--output_report",
                                    str(
                                        predictions_csv.with_name(
                                            f"{predictions_csv.stem}_eval_summary.txt"
                                        )
                                    ),
                                ]
                            )
                        if args.no_incidence_reports:
                            metrics_cmd.append("--no_incidence_reports")
                        if _run_subprocess(metrics_cmd, env) != 0:
                            raise RuntimeError("llm_eval_metrics.py failed")

                        if not predictions_csv.is_file():
                            raise RuntimeError(f"Missing predictions: {predictions_csv}")
                        ran_ok = True

                except Exception as e:
                    reason = str(e)
                    if skip_on_failure:
                        _record_skip(slug, model_id, reason, quant)
                        _stop_process(vllm_proc)
                        vllm_proc = None
                        continue
                    raise

                predictions_csv = (
                    Path(args.output_csv).resolve()
                    if args.output_csv
                    else _resolve_predictions_csv(model_dir)
                )
                if not ran_ok or not predictions_csv.is_file():
                    reason = "no predictions produced"
                    if skip_on_failure:
                        _record_skip(slug, model_id, reason, quant)
                        continue
                    raise RuntimeError(f"{slug}: {reason}")

                summary = summarize_predictions_csv(str(predictions_csv), mode=args.mode)
                row = {
                    "model_slug": slug,
                    "model_id": model_id,
                    **summary,
                }
                row.setdefault("lat_detection_inc1_pct", float("nan"))
                row.setdefault("lat_detection_inc_ge2_pct", float("nan"))
                comparison_rows.append(row)
                suite_paths = _suite_model_paths(model_dir)
                incidence_reports = {
                    incidence_mode: str(
                        suite_paths["results_dir"]
                        / incidence_report_filename(incidence_mode, args.mode)
                    )
                    for incidence_mode in incidence_modes_for_eval_mode(args.mode)
                }
                manifest_models.append(
                    {
                        "slug": slug,
                        "model_id": model_id,
                        "quantization": quant,
                        "status": "ok",
                        "predictions_csv": str(predictions_csv),
                        "report_txt": str(report_txt),
                        "incidence_reports": incidence_reports,
                        "confusion_plot": str(suite_paths["confusion_plot"]),
                        "cuda_visible_devices": model_gpu,
                    }
                )
    finally:
        _stop_process(vllm_proc)

    if args.bert_predictions:
        bert_summary = summarize_predictions_csv(args.bert_predictions, mode=args.mode)
        bert_row = {
            "model_slug": args.bert_label,
            "model_id": "(fine-tuned BERT)",
            **bert_summary,
        }
        bert_row.setdefault("lat_detection_inc1_pct", float("nan"))
        bert_row.setdefault("lat_detection_inc_ge2_pct", float("nan"))
        comparison_rows.insert(0, bert_row)

    suite_meta = {
        "suite_name": args.suite_name,
        "eval_csv": str(Path(args.eval_csv).resolve()),
        "mode": args.mode,
        "prompt": args.prompt,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_url": args.base_url,
        "manage_vllm": manage_vllm,
        "models_json": str(Path(args.models_json).resolve()),
        "limit": args.limit,
        "cuda_visible_devices": args.gpu,
        "skip_on_failure": skip_on_failure,
    }
    manifest = {
        "suite": suite_meta,
        "models": manifest_models,
        "skipped": skipped_models,
    }
    (suite_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    _write_comparison_table(
        comparison_rows,
        suite_root / "comparison_table.csv",
        suite_root / "comparison_table.md",
        suite_meta,
    )
    _write_comparison_by_error_type(
        comparison_rows,
        suite_root / "comparison_by_error_type.csv",
    )
    if skipped_models:
        print("\nSkipped models:")
        for item in skipped_models:
            print(f"  - {item['slug']}: {item['reason']}")
    if not comparison_rows and not args.bert_predictions:
        print("\nWARNING: No successful LLM runs; comparison table is empty.")
    print(f"\nSuite complete: {suite_root}")


if __name__ == "__main__":
    main()
