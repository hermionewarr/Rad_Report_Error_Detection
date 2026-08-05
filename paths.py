"""Canonical paths and per-run layout for this project."""
from __future__ import annotations

import os
import re
import shlex
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

BERT_MODEL_ROOT = Path(__file__).resolve().parent
SRC_DIR = BERT_MODEL_ROOT / "src"
DATA_DIR = BERT_MODEL_ROOT / "data"
EVAL_DATA_DIR = DATA_DIR / "eval_datasets"
TRAINED_MODELS_DIR = BERT_MODEL_ROOT / "trained_models"
RUNS_DIR = TRAINED_MODELS_DIR / "runs"
RUNS_PRETRAINING_DIR = RUNS_DIR / "pretraining"
RUNS_LLM_BASELINE_DIR = RUNS_DIR / "llm_baseline"
RUNS_BERT_BINARY_DIR = RUNS_DIR / "bert_binary"
RUNS_BERT_MULTICLASS_DIR = RUNS_DIR / "bert_multiclass"
LEGACY_OUTPUT_DIR = TRAINED_MODELS_DIR / "output"
NO_FINDINGS_DIR = BERT_MODEL_ROOT / "no_findings_analysis"
ANALYSIS_FIGURES_DIR = BERT_MODEL_ROOT / "analysis_figures"
# Project root (same as BERT_MODEL_ROOT; kept as a clearer alias).
PROJECT_ROOT = BERT_MODEL_ROOT
# Per-run stdout/stderr capture under logs/output.log.
RUN_OUTPUT_LOG = "output.log"


def _env_path(var: str, default: Path) -> Path:
    """Resolve an optional env override; otherwise use *default* under this repo."""
    raw = os.environ.get(var)
    if raw:
        return Path(raw).expanduser().resolve()
    return default


# ---- User-supplied data (override with env or place files under data/) ----
# REPORTS_CSV: master CSV with Findings, Conclusion, and optional split column.
DEFAULT_REPORTS_CSV = _env_path("REPORTS_CSV", DATA_DIR / "reports.csv")
# TOKENIZER_PATH: HuggingFace tokenizer directory (or name loadable by AutoTokenizer).
DEFAULT_TOKENIZER_PATH = str(_env_path("TOKENIZER_PATH", DATA_DIR / "tokenizer"))
# CHECKPOINTS_ROOT: optional extra tree to search for pretrained weights.
# default: only this repo's trained_models/


@dataclass(frozen=True)
class RunLayout:
    """trained_models/runs/<run_name>/{model,results,plots,logs}/"""

    root: Path
    model: Path
    results: Path
    plots: Path
    logs: Path

    def result_path(self, filename: str) -> Path:
        return self.results / filename

    def plot_path(self, filename: str) -> Path:
        return self.plots / filename

    def log_path(self, filename: str) -> Path:
        return self.logs / filename


def data_path(name: str) -> str:
    return str(DATA_DIR / name)


def eval_data_path(name: str) -> str:
    return str(EVAL_DATA_DIR / name)


def pool_data_path(name: str) -> str:
    """Derived pool CSVs stored directly under data/ (not eval_datasets/)."""
    return str(DATA_DIR / name)


def resolve_existing_csv(name: str) -> Optional[Path]:
    """Return first existing path for a CSV in data/ or data/eval_datasets/."""
    for base in (DATA_DIR, EVAL_DATA_DIR):
        p = base / name
        if p.is_file():
            return p
    return None


def resolve_existing_csv_path(name: str) -> str:
    """Like resolve_existing_csv but returns the standard write location if missing."""
    found = resolve_existing_csv(name)
    if found is not None:
        return str(found)
    if name.startswith("mid_hard_negative") or name.startswith("mid_hard_eval_pool"):
        return pool_data_path(name)
    return eval_data_path(name)


# Canonical init checkpoint run names (weights under runs/pretraining/<name>/model/).
SUPPORTED_SEQ_LENGTHS = (512, 600, 768)
DEFAULT_SEQ_LENGTH = 600


def _model_size_suffix(model_size: str = "small") -> str:
    from bert_architectures import model_size_suffix

    return model_size_suffix(model_size)


def init_cls_checkpoint_name(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> str:
    if seq_length not in SUPPORTED_SEQ_LENGTHS:
        raise ValueError(
            f"seq_length must be one of {SUPPORTED_SEQ_LENGTHS}, got {seq_length!r}"
        )
    suffix = _model_size_suffix(model_size)
    return f"bert_cls_classification_mxln{seq_length}{suffix}"


def init_cls_checkpoint_path(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> str:
    return checkpoint_path(
        init_cls_checkpoint_name(seq_length, model_size=model_size),
        require_exists=False,
    )


def init_pretrain_checkpoint_name(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> str:
    if seq_length not in SUPPORTED_SEQ_LENGTHS:
        raise ValueError(
            f"seq_length must be one of {SUPPORTED_SEQ_LENGTHS}, got {seq_length!r}"
        )
    suffix = _model_size_suffix(model_size)
    return f"bert-pretrained_mxlen_{seq_length}{suffix}"


def init_pretrain_checkpoint_path(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> str:
    return checkpoint_path(
        init_pretrain_checkpoint_name(seq_length, model_size=model_size),
        require_exists=False,
    )


def init_pretrain_model_dir(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> Path:
    return RUNS_PRETRAINING_DIR / init_pretrain_checkpoint_name(seq_length, model_size=model_size) / "model"


def init_cls_model_dir(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    model_size: str = "small",
) -> Path:
    return RUNS_PRETRAINING_DIR / init_cls_checkpoint_name(seq_length, model_size=model_size) / "model"


def init_model_checkpoint_path(
    seq_length: int = DEFAULT_SEQ_LENGTH,
    *,
    wo_cls_init: bool = False,
    model_size: str = "small",
) -> str:
    if wo_cls_init:
        return init_pretrain_checkpoint_path(seq_length, model_size=model_size)
    return init_cls_checkpoint_path(seq_length, model_size=model_size)


INIT_CLS_CHECKPOINT = init_cls_checkpoint_name(DEFAULT_SEQ_LENGTH)
INIT_CONTRASTIVE_CHECKPOINT = "bert-contrastive_bs128"


def runs_base_dir(run_category: Optional[str] = None) -> Path:
    if not run_category:
        return RUNS_DIR
    mapping = {
        "pretraining": RUNS_PRETRAINING_DIR,
        "llm_baseline": RUNS_LLM_BASELINE_DIR,
        "bert_binary": RUNS_BERT_BINARY_DIR,
        "bert_multiclass": RUNS_BERT_MULTICLASS_DIR,
    }
    try:
        return mapping[run_category]
    except KeyError as exc:
        raise ValueError(
            f"Unknown run_category={run_category!r}. "
            "Expected one of: pretraining, llm_baseline, bert_binary, bert_multiclass."
        ) from exc


def ensure_named_run_layout(name: str, run_category: Optional[str] = None) -> RunLayout:
    """Ensure a stable named run has model/, results/, plots/, logs/."""
    root = runs_base_dir(run_category) / name
    layout = RunLayout(
        root=root,
        model=root / "model",
        results=root / "results",
        plots=root / "plots",
        logs=root / "logs",
    )
    for d in (layout.model, layout.results, layout.plots, layout.logs):
        d.mkdir(parents=True, exist_ok=True)
    return layout


def _external_trained_models_dirs() -> list[Path]:
    """Optional extra weight trees (CHECKPOINTS_ROOT or REPORT_INCONSISTENCY_CHECKPOINTS_ROOT)."""
    dirs: list[Path] = []
    for var in ("CHECKPOINTS_ROOT", "REPORT_INCONSISTENCY_CHECKPOINTS_ROOT"):
        raw = os.environ.get(var)
        if raw:
            dirs.append(Path(raw).expanduser().resolve())
    return dirs


def _checkpoint_model_dirs(name: str) -> list[Path]:
    """Candidate model/ dirs for a named checkpoint, most preferred first."""
    candidates: list[Path] = []
    for category in ("pretraining", "bert_binary", "bert_multiclass"):
        candidates.append(runs_base_dir(category) / name / "model")
    candidates.append(RUNS_DIR / name / "model")  # legacy flat runs/<name>/model
    symlink = BERT_MODEL_ROOT / name
    if symlink.is_symlink():
        candidates.append(symlink.resolve())
    candidates.append(TRAINED_MODELS_DIR / name)
    for external in _external_trained_models_dirs():
        for category in ("pretraining", "bert_binary", "bert_multiclass"):
            candidates.append(external / "runs" / category / name / "model")
        candidates.append(external / "runs" / name / "model")
        candidates.append(external / name)
    return candidates


def resolve_checkpoint(name: str) -> Path:
    """Resolve a named checkpoint to a directory containing config + weights."""
    for model_dir in _checkpoint_model_dirs(name):
        if model_dir.is_dir() and (model_dir / "config.json").exists():
            has_weights = any(
                (model_dir / fname).exists()
                for fname in (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "tf_model.h5",
                )
            )
            if has_weights:
                return model_dir
    searched = "\n  ".join(str(p) for p in _checkpoint_model_dirs(name))
    raise FileNotFoundError(
        f"Checkpoint {name!r} not found (need config.json + model weights). Searched:\n  {searched}"
    )


def expected_checkpoint_dir(name: str) -> Path:
    """Canonical on-disk location for a named checkpoint (may not exist yet)."""
    return RUNS_PRETRAINING_DIR / name / "model"


def checkpoint_path(name: str, *, require_exists: bool = True) -> str:
    """Return path to checkpoint weights.

    If require_exists is False (defaults used at module import), return the
    preferred path even when weights have not been copied into this repo yet.
    """
    try:
        return str(resolve_checkpoint(name))
    except FileNotFoundError:
        if require_exists:
            raise
        return str(expected_checkpoint_dir(name))


def run_dir(run_name: str, *subpaths: str, run_category: Optional[str] = None) -> Path:
    path = runs_base_dir(run_category) / run_name
    for part in subpaths:
        path = path / part
    return path


def resolve_under_root(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return BERT_MODEL_ROOT / p


def _slugify(parts: Iterable[str], max_len: int = 72) -> str:
    raw = "_".join(p for p in parts if p)
    out = []
    for ch in raw:
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch in " ./":
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len] if slug else "run"


_DATE_PREFIX_RE = re.compile(r"^\d{8}_")


def make_run_name(script_id: str, tags: Sequence[str] = (), run_name: Optional[str] = None) -> str:
    if run_name:
        if _DATE_PREFIX_RE.match(run_name):
            return run_name
        date_prefix = datetime.now().strftime("%Y%m%d")
        return f"{date_prefix}_{run_name}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_slug = _slugify(tags)
    return f"{ts}_{script_id}" + (f"_{tag_slug}" if tag_slug else "")


def create_run(
    script_id: str,
    *,
    tags: Sequence[str] = (),
    run_name: Optional[str] = None,
    run_category: Optional[str] = None,
    command_stem: str = "command",
    argv: Optional[Sequence[str]] = None,
) -> RunLayout:
    name = make_run_name(script_id, tags=tags, run_name=run_name)
    root = run_dir(name, run_category=run_category)
    layout = RunLayout(
        root=root,
        model=root / "model",
        results=root / "results",
        plots=root / "plots",
        logs=root / "logs",
    )
    for d in (layout.model, layout.results, layout.plots, layout.logs):
        d.mkdir(parents=True, exist_ok=True)
    if argv is not None:
        save_run_command(layout.root, argv, stem=command_stem)
    return layout


def save_run_command(run_root: Path, argv: Sequence[str], *, stem: str = "command") -> None:
    argv_list = list(argv)
    cmd = " ".join(shlex.quote(a) for a in argv_list)
    (run_root / f"{stem}.txt").write_text(cmd + "\n", encoding="utf-8")
    meta = run_root / f"{stem}_argv.txt"
    meta.write_text("\n".join(argv_list) + "\n", encoding="utf-8")


def current_argv() -> list[str]:
    return [Path(sys.argv[0]).name, *sys.argv[1:]]


def setup_run(
    script_id: str,
    *,
    tags: Sequence[str] = (),
    run_name: Optional[str] = None,
    run_category: Optional[str] = None,
    output_model_dir: Optional[str | Path] = None,
    command_stem: str = "command",
    argv: Optional[Sequence[str]] = None,
) -> RunLayout:
    """Create a timestamped run, or wrap an explicit --output_model_dir path."""
    argv = list(argv) if argv is not None else current_argv()
    if output_model_dir:
        model = Path(output_model_dir)
        if model.name == "model":
            root = model.parent
        else:
            root = model
            model = root / "model"
        layout = RunLayout(
            root=root,
            model=model,
            results=root / "results",
            plots=root / "plots",
            logs=root / "logs",
        )
        for d in (layout.model, layout.results, layout.plots, layout.logs):
            d.mkdir(parents=True, exist_ok=True)
        save_run_command(layout.root, argv, stem=command_stem)
        return layout
    return create_run(
        script_id,
        tags=tags,
        run_name=run_name,
        run_category=run_category,
        argv=argv,
        command_stem=command_stem,
    )


def default_result_path(
    layout: RunLayout,
    filename: str,
    explicit: Optional[str],
) -> str:
    if explicit:
        return explicit
    return str(layout.result_path(filename))


def default_plot_path(
    layout: RunLayout,
    filename: str,
    explicit: Optional[str],
) -> str:
    if explicit:
        return explicit
    return str(layout.plot_path(filename))


def resolve_run_layout_from_artifact(path: str | Path) -> Optional[RunLayout]:
    """Infer a RunLayout when *path* lives under .../<run>/{model,results,plots,logs}/."""
    p = Path(path).resolve()
    for parent in (p, *p.parents):
        results = parent / "results"
        model = parent / "model"
        if results.is_dir() and model.is_dir():
            return RunLayout(
                root=parent,
                model=model,
                results=results,
                plots=parent / "plots",
                logs=parent / "logs",
            )
    return None


def infer_run_root_from_predictions(predictions_csv: str | Path) -> Optional[Path]:
    """Run root for eval artifacts; prefers the run that owns predictions.csv."""
    layout = resolve_run_layout_from_artifact(predictions_csv)
    if layout is not None:
        return layout.root
    p = Path(predictions_csv).resolve()
    if p.parent.name == "results":
        return p.parent.parent
    return None


class _TeeStream:
    """Write to terminal and log file (line-buffered)."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._stream.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):
        return self._stream.fileno()


@contextmanager
def tee_run_output(log_path: str | Path):
    """Mirror stdout/stderr to a run log file while still printing to the terminal."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, log_file)
        sys.stderr = _TeeStream(old_stderr, log_file)
        print(f"Logging stdout/stderr to: {log_path.resolve()}", flush=True)
        try:
            yield log_path
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
