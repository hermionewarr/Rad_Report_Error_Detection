#!/usr/bin/env python3
"""Plot BERT scaling metrics, BERT vs LLM comparisons, and clinical-specificity ladders (v6 eval)."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from transformers import BertModel

REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_ROOT = REPO_ROOT
SCRIPTS_DIR = BERT_ROOT / "scripts"
RUNS_DIR = BERT_ROOT / "trained_models" / "runs" / "bert_binary"
LLM_BASELINE_DIR = BERT_ROOT / "trained_models" / "runs" / "llm_baseline"
FIGURES_DIR = BERT_ROOT / "figures"

import sys

sys.path.insert(0, str(BERT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
from bert_architectures import bert_mlm_config  # noqa: E402
import plot_clinical_specificity_ladder as ladder_plot  # noqa: E402

VOCAB_SIZE = 10_990
SEQ_LENGTH = 600

# Binary hard-single cls600 runs, one per layer count (4 / 6 / 8 / 12).
# June 29 mlmcls-preclshead sweep (same family as clinical-specificity ladder).
DEFAULT_RUNS: dict[int, str] = {
    4: "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-4",
    6: "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-6",
    8: "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-8L",
    12: "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-12L",
}

MODEL_SIZE_BY_LAYERS = {
    4: "small",
    6: "big",
    8: "8L",
    12: "12L",
}

DEFAULT_LLM_SUITE = "20260626_llm-suite__p3"
# Best prompt per model: Gemma p3; Qwen/Llama p8 (split across dated suite dirs).
DEFAULT_LLM_SUITES: dict[str, str] = {
    "gemma-3-27b-gguf": "20260626_llm-suite__p3",
    "qwen3-32b-gguf": "20260629_llm-suite__p8",
    "llama-3.3-70b": "20260630_llm-suite__p8",
}
DEFAULT_LLMS: tuple[tuple[str, int, str], ...] = (
    ("Gemma-27B", 27, "gemma-3-27b-gguf"),
    ("Qwen-32B", 32, "qwen3-32b-gguf"),
    ("Llama-70B", 70, "llama-3.3-70b"),
)

METRIC_ROW_LABELS = ("Bal-bal acc", "TPR", "TNR")

ACC_RE = re.compile(r"overall accuracy:\s+([\d.]+)%")
BAL_BAL_RE = re.compile(r"balanced-balanced acc:\s+([\d.]+)%")
TPR_RE = re.compile(r"^inconsistent recall:\s+\d+\s+\(([\d.]+)%\)", re.MULTILINE)
TNR_RE = re.compile(r"^consistent recall:\s+\d+\s+\(([\d.]+)%\)", re.MULTILINE)
FPR_RE = re.compile(r"^false alarms:\s+\d+\s+\(([\d.]+)%\)", re.MULTILINE)
MACRO_ERR_RE = re.compile(r"^macro error recall:\s+([\d.]+)%", re.MULTILINE)
BY_TYPE_RECALL_RE = re.compile(
    r"^\s+(?!consistent\b)(\S+)\s+\d+\s+\d+\s+\d+\s+[\d.]+%\s+([\d.]+)%",
    re.MULTILINE,
)
LOG_ACC_RE = re.compile(r"\[overall\]\s+loss=[\d.]+\s+acc=([\d.]+)")
EVAL_CSV_RE = re.compile(r"^Eval CSV:\s+(.+)$", re.MULTILINE)
EVAL_MODE_RE = re.compile(r"^Eval mode:\s+(\S+)$", re.MULTILINE)


@dataclass
class EvalMetrics:
    accuracy: float | None = None
    bal_bal_acc: float | None = None
    tpr: float | None = None
    tnr: float | None = None
    fpr: float | None = None
    source: str = ""
    pending: bool = False


@dataclass
class ModelPoint:
    layers: int
    model_size: str
    run_name: str
    num_params: int
    metrics: EvalMetrics


@dataclass
class HeatmapColumn:
    name: str
    param_label: str
    num_params: int
    metrics: EvalMetrics


def count_parameters(model_size: str) -> int:
    config = bert_mlm_config(VOCAB_SIZE, SEQ_LENGTH, model_size)
    bert = BertModel(config, add_pooling_layer=False)
    head = nn.Linear(config.hidden_size, 1)
    return sum(p.numel() for p in bert.parameters()) + sum(p.numel() for p in head.parameters())


def _parse_eval_summary(path: Path) -> tuple[EvalMetrics, bool]:
    """Return parsed metrics and whether this is a v6 mode=all summary."""
    text = path.read_text(encoding="utf-8")
    csv_match = EVAL_CSV_RE.search(text)
    mode_match = EVAL_MODE_RE.search(text)
    eval_csv = csv_match.group(1) if csv_match else ""
    eval_mode = mode_match.group(1) if mode_match else ""
    is_v6_all = "v6" in eval_csv and eval_mode == "all"

    label = f"{path.parent.name}/eval_summary.txt"
    if eval_mode:
        label += f" (mode={eval_mode})"

    def _pct(pattern: re.Pattern[str]) -> float | None:
        match = pattern.search(text)
        return float(match.group(1)) if match else None

    tpr = _pct(TPR_RE)
    tnr = _pct(TNR_RE)
    fpr = _pct(FPR_RE)
    if fpr is None and tnr is not None:
        fpr = round(100.0 - tnr, 1)
    bal_bal = _pct(BAL_BAL_RE)
    if bal_bal is None and tnr is not None:
        macro_err = _pct(MACRO_ERR_RE)
        if macro_err is None:
            recalls = [float(v) for _, v in BY_TYPE_RECALL_RE.findall(text)]
            if recalls:
                macro_err = sum(recalls) / len(recalls)
        if macro_err is not None:
            bal_bal = 0.5 * tnr + 0.5 * macro_err

    metrics = EvalMetrics(
        accuracy=_pct(ACC_RE),
        bal_bal_acc=bal_bal,
        tpr=tpr,
        tnr=tnr,
        fpr=fpr,
        source=label,
    )
    return metrics, is_v6_all


def _parse_v6eval_log(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8")
    if "mode=all" not in text:
        return None
    match = LOG_ACC_RE.search(text)
    if not match:
        return None
    return round(float(match.group(1)) * 100.0, 1)


def load_eval_metrics(run_name: str, *, require_v6_all: bool) -> EvalMetrics:
    run_dir = RUNS_DIR / run_name
    candidates: list[tuple[int, EvalMetrics]] = []

    for summary_path in (
        run_dir.parent / f"{run_name}-v6eval" / "results" / "eval_summary.txt",
        run_dir / "results" / "eval_summary.txt",
    ):
        if not summary_path.exists():
            continue
        metrics, is_v6_all = _parse_eval_summary(summary_path)
        if metrics.accuracy is not None:
            priority = 3 if is_v6_all else 0
            candidates.append((priority, metrics))

    v6eval_log = run_dir.parent / f"{run_name}-v6eval" / "logs" / "output.log"
    if v6eval_log.exists():
        acc = _parse_v6eval_log(v6eval_log)
        if acc is not None:
            interim = EvalMetrics(
                accuracy=acc,
                source=f"{run_name}-v6eval/logs/output.log (interim)",
            )
            candidates.append((2, interim))

    if not candidates:
        return EvalMetrics(source="missing", pending=True)

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_priority, best = candidates[0]
    best.pending = require_v6_all and best_priority < 3
    return best


def load_llm_columns(
    llms: tuple[tuple[str, int, str], ...],
    *,
    llm_suite: str | None = None,
    llm_suites: dict[str, str] | None = None,
) -> list[HeatmapColumn]:
    suites = llm_suites or DEFAULT_LLM_SUITES
    columns: list[HeatmapColumn] = []
    for name, params_b, slug in llms:
        suite = llm_suite or suites.get(slug, DEFAULT_LLM_SUITE)
        summary = LLM_BASELINE_DIR / suite / "models" / slug / "results" / "eval_summary.txt"
        if not summary.exists():
            raise FileNotFoundError(f"Missing LLM eval summary: {summary}")
        metrics, _ = _parse_eval_summary(summary)
        if metrics.bal_bal_acc is None or metrics.tpr is None or metrics.tnr is None:
            raise SystemExit(f"Incomplete metrics in {summary}")
        columns.append(
            HeatmapColumn(
                name=name,
                param_label=f"{params_b}B",
                num_params=params_b * 1_000_000_000,
                metrics=metrics,
            )
        )
    return columns


def _llm_suite_footnote(llm_suite: str | None = None) -> str:
    if llm_suite:
        return llm_suite
    parts = []
    for name, _, slug in DEFAULT_LLMS:
        suite = DEFAULT_LLM_SUITES.get(slug, DEFAULT_LLM_SUITE)
        prompt = suite.rsplit("__", 1)[-1] if "__" in suite else suite
        parts.append(f"{name.split('-')[0]} {prompt}")
    return ", ".join(parts)

def _format_params_m(num_params: int) -> str:
    if num_params >= 1_000_000:
        value = num_params / 1_000_000
        return f"{value:.0f}M" if value >= 10 else f"{value:.1f}M"
    return f"{num_params:,}"


def _metrics_matrix(columns: list[HeatmapColumn]) -> np.ndarray:
    return np.array(
        [
            [col.metrics.bal_bal_acc for col in columns],
            [col.metrics.tpr for col in columns],
            [col.metrics.tnr for col in columns],
        ],
        dtype=float,
    )


def _draw_heatmap_panel(
    ax: plt.Axes,
    columns: list[HeatmapColumn],
    *,
    data: np.ndarray,
    vmin: float,
    vmax: float,
    cmap: str,
    show_ylabel: bool,
) -> object:
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([col.param_label for col in columns], fontsize=9)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlabel("Parameters", fontsize=9, labelpad=6)

    ax.set_yticks(range(len(METRIC_ROW_LABELS)))
    if show_ylabel:
        ax.set_yticklabels(METRIC_ROW_LABELS, fontsize=9)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    for col_idx, col in enumerate(columns):
        ax.text(
            col_idx,
            -0.72,
            col.name,
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=25,
            transform=ax.get_xaxis_transform(),
        )

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            value = data[row_idx, col_idx]
            text_color = "white" if value < 78 else "#111111"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                fontweight="medium",
            )

    ax.tick_params(top=False, bottom=True, left=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _add_axis_break_marks(left_ax: plt.Axes, right_ax: plt.Axes) -> None:
    kwargs = dict(color="#444444", clip_on=False, linewidth=1.0)
    left_ax.plot((1.02, 1.08), (-0.14, -0.02), transform=left_ax.transAxes, **kwargs)
    left_ax.plot((1.02, 1.08), (1.02, 1.12), transform=left_ax.transAxes, **kwargs)
    right_ax.plot((-0.08, -0.02), (-0.14, -0.02), transform=right_ax.transAxes, **kwargs)
    right_ax.plot((-0.08, -0.02), (1.02, 1.12), transform=right_ax.transAxes, **kwargs)


def plot_combined_heatmap(
    bert_columns: list[HeatmapColumn],
    llm_columns: list[HeatmapColumn],
    *,
    output_path: Path,
    title: str = "Inconsistency detection — v6 eval",
) -> None:
    bert_data = _metrics_matrix(bert_columns)
    llm_data = _metrics_matrix(llm_columns)
    all_values = np.concatenate([bert_data.ravel(), llm_data.ravel()])
    vmin = max(50.0, float(np.floor(all_values.min() - 2)))
    vmax = 100.0
    cmap = "RdYlGn"

    fig = plt.figure(figsize=(10.5, 3.6))
    gs = GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[len(bert_columns), 0.35, len(llm_columns)],
        wspace=0.06,
    )
    ax_bert = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_llm = fig.add_subplot(gs[0, 2])

    im = _draw_heatmap_panel(
        ax_bert,
        bert_columns,
        data=bert_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_ylabel=True,
    )
    _draw_heatmap_panel(
        ax_llm,
        llm_columns,
        data=llm_data,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        show_ylabel=False,
    )

    ax_gap.axis("off")
    ax_gap.text(
        0.5,
        0.5,
        "···",
        ha="center",
        va="center",
        fontsize=18,
        color="#666666",
        transform=ax_gap.transAxes,
    )

    _add_axis_break_marks(ax_bert, ax_llm)

    ax_bert.set_title("Fine-tuned BERT (cls600)", fontsize=10, pad=22)
    ax_llm.set_title("LLM zero-shot (prompt 12)", fontsize=10, pad=22)

    cbar = fig.colorbar(im, ax=[ax_bert, ax_llm], fraction=0.025, pad=0.02)
    cbar.set_label("Rate (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "BERT: hard training, eval mode=all  ·  LLM: prompt 12, eval mode=hard (parse_ok rows)",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )

    fig.subplots_adjust(top=0.82, bottom=0.24, left=0.07, right=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def _log_axis_formatter_params(value: float, _pos: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.0f}B"
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:.0f}M" if scaled >= 10 else f"{scaled:.1f}M"
    return f"{value:.0g}"


def _label_offsets(columns: list[HeatmapColumn]) -> list[tuple[float, float]]:
    """Stagger labels when neighbouring points are close in x and y."""
    offsets: list[tuple[float, float]] = [(0.0, 8.0)] * len(columns)
    for i in range(len(columns) - 1):
        c1, c2 = columns[i], columns[i + 1]
        log_gap = abs(np.log10(c2.num_params) - np.log10(c1.num_params))
        y_gap = abs(c2.metrics.bal_bal_acc - c1.metrics.bal_bal_acc)
        if log_gap < 0.15 and y_gap < 4.0:
            offsets[i] = (-10.0, 10.0)
            offsets[i + 1] = (10.0, -13.0)
    return offsets


def _annotate_log_point(
    ax: plt.Axes,
    col: HeatmapColumn,
    *,
    color: str,
    xytext: tuple[float, float] = (0.0, 8.0),
) -> None:
    m = col.metrics
    ax.scatter(
        [col.num_params],
        [m.bal_bal_acc],
        s=70,
        color=color,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.annotate(
        col.name,
        (col.num_params, m.bal_bal_acc),
        textcoords="offset points",
        xytext=xytext,
        ha="center",
        fontsize=7.5,
        color="#333333",
    )


DETECTION_COLOR = "#55A868"
FALSE_ALARM_COLOR = "#C44E52"


def plot_detection_vs_false_alarm(
    bert_columns: list[HeatmapColumn],
    llm_columns: list[HeatmapColumn],
    *,
    output_path: Path,
    title: str = "Detection recall vs false alarm rate",
    llm_suite_note: str | None = None,
) -> None:
    """Grouped bars: TPR (detection) and FPR (false alarm) per model."""
    columns = bert_columns + llm_columns
    for col in columns:
        if col.metrics.tpr is None or col.metrics.fpr is None:
            raise SystemExit(f"Incomplete TPR/FPR in metrics for {col.name}")

    labels = [col.name for col in columns]
    detection = [col.metrics.tpr for col in columns]
    false_alarm = [col.metrics.fpr for col in columns]

    n_bert = len(bert_columns)
    n_models = len(columns)
    gap = 0.6
    x = np.arange(n_models, dtype=float)
    if n_bert < n_models:
        x[n_bert:] += gap

    width = 0.34
    fig, ax = plt.subplots(figsize=(11.0, 4.9))

    det_bars = ax.bar(
        x - width / 2,
        detection,
        width,
        label="Detection recall (TPR)",
        color=DETECTION_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    fa_bars = ax.bar(
        x + width / 2,
        false_alarm,
        width,
        label="False alarm rate (FPR)",
        color=FALSE_ALARM_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    for bars in (det_bars, fa_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    if n_bert < n_models:
        divider = (x[n_bert - 1] + x[n_bert]) / 2.0
        ax.axvline(divider, color="#CCCCCC", linestyle="--", linewidth=1.0, zorder=1)
        ax.text(
            (x[0] + x[n_bert - 1]) / 2.0,
            0.985,
            "Domain-specific BERT",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color="#4C72B0",
            fontweight="medium",
            clip_on=False,
        )
        ax.text(
            (x[n_bert] + x[-1]) / 2.0,
            0.985,
            "Generic-domain LLMs",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=11,
            color="#C4713B",
            fontweight="medium",
            clip_on=False,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Rate (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.legend(
        loc="upper right",
        fontsize=9.5,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.3,
        bbox_to_anchor=(0.995, 0.955),
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.28, zorder=0)

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(
        0.5,
        0.01,
        "BERT: hard training (20260629 mlmcls-preclshead), eval mode=all  ·  "
        f"LLM: {llm_suite_note or _llm_suite_footnote()}, eval mode=hard (parse_ok rows)",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )

    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.07, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def plot_combined_log_scale(
    bert_columns: list[HeatmapColumn],
    llm_columns: list[HeatmapColumn],
    *,
    output_path: Path,
    title: str = "Inconsistency detection — v6 eval",
) -> None:
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(9.2, 4.2))

    all_columns = bert_columns + llm_columns
    all_y = [c.metrics.bal_bal_acc for c in all_columns]
    ymin = max(55.0, min(all_y) - 4)
    ymax = min(100.0, max(all_y) + 3)

    for columns, color, legend_label in (
        (bert_columns, "#4C72B0", "Fine-tuned BERT (cls600)"),
        (llm_columns, "#C4713B", "LLM zero-shot (prompt 12)"),
    ):
        xs = [c.num_params for c in columns]
        ys = [c.metrics.bal_bal_acc for c in columns]
        ax.plot(xs, ys, color="#AAAAAA", linewidth=1.1, zorder=1)
        for col, offset in zip(columns, _label_offsets(columns)):
            _annotate_log_point(ax, col, color=color, xytext=offset)
        ax.scatter([], [], s=70, color=color, edgecolors="white", linewidths=0.8, label=legend_label)

    all_x = [c.num_params for c in all_columns]
    ax.set_xscale("log")
    ax.set_xlim(min(all_x) * 0.75, max(all_x) * 1.25)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Bal-balanced accuracy (%)")
    ax.set_xlabel("Parameters", fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_log_axis_formatter_params))
    ax.grid(True, which="both", linestyle="--", alpha=0.28)
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(
        0.5,
        0.01,
        "BERT: hard training, eval mode=all  ·  LLM: prompt 12, eval mode=hard (parse_ok rows)",
        ha="center",
        fontsize=7.5,
        color="#555555",
    )

    fig.subplots_adjust(top=0.88, bottom=0.16, left=0.08, right=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def bert_plot_columns(plottable: list[ModelPoint]) -> list[HeatmapColumn]:
    ready = _points_with_class_metrics(plottable)
    return [
        HeatmapColumn(
            name=f"{point.layers}L",
            param_label=_format_params_m(point.num_params),
            num_params=point.num_params,
            metrics=point.metrics,
        )
        for point in ready
    ]


def collect_points(
    runs: dict[int, str],
    *,
    require_v6_all: bool,
) -> list[ModelPoint]:
    points: list[ModelPoint] = []
    for layers in sorted(runs):
        run_name = runs[layers]
        model_size = MODEL_SIZE_BY_LAYERS[layers]
        metrics = load_eval_metrics(run_name, require_v6_all=require_v6_all)
        points.append(
            ModelPoint(
                layers=layers,
                model_size=model_size,
                run_name=run_name,
                num_params=count_parameters(model_size),
                metrics=metrics,
            )
        )
    return points


def _points_with_class_metrics(points: list[ModelPoint]) -> list[ModelPoint]:
    return [
        p
        for p in points
        if p.metrics.bal_bal_acc is not None
        and p.metrics.tpr is not None
        and p.metrics.tnr is not None
    ]


METRIC_SERIES: tuple[tuple[str, str, str], ...] = (
    ("Bal-bal acc", "bal_bal_acc", "#4C72B0"),
    ("TPR", "tpr", "#55A868"),
    ("TNR", "tnr", "#C44E52"),
)


def plot_scaling_curve(
    plottable: list[ModelPoint],
    all_points: list[ModelPoint],
    *,
    output_path: Path,
    title: str = "BERT inconsistency detector: size vs performance",
) -> None:
    ready = _points_with_class_metrics(plottable)
    if not ready:
        raise SystemExit(
            "No bal-bal / TPR / TNR values found. Run v6 evals first or pass --allow-partial."
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [p.num_params / 1e6 for p in ready]
    all_ys: list[float] = []

    for label, attr, color in METRIC_SERIES:
        ys = [getattr(p.metrics, attr) for p in ready]
        all_ys.extend(ys)
        ax.plot(xs, ys, color=color, linewidth=1.4, marker="o", markersize=7, label=label)

    for point in ready:
        ax.annotate(
            f"{point.layers}L",
            (point.num_params / 1e6, point.metrics.bal_bal_acc),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="#333333",
        )

    pending = [
        p
        for p in all_points
        if p.metrics.pending
        or p.metrics.bal_bal_acc is None
        or p.metrics.tpr is None
        or p.metrics.tnr is None
    ]
    if pending:
        missing_layers = ", ".join(f"{p.layers}L" for p in pending)
        ax.text(
            0.02,
            0.02,
            f"Pending v6 (mode=all): {missing_layers}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#8B0000",
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
        )

    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("Rate (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(left=0)
    ymin = max(0.0, min(all_ys) - 3)
    ymax = min(100.0, max(all_ys) + 2)
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def plot_metrics_bars(
    plottable: list[ModelPoint],
    *,
    output_path: Path,
    title: str = "BERT inconsistency detector — class-balanced metrics",
) -> None:
    ready = _points_with_class_metrics(plottable)
    if not ready:
        raise SystemExit("No bal-bal / TPR / TNR values found in eval summaries.")

    labels = [f"{p.layers}L" for p in ready]
    metric_names = [name for name, _, _ in METRIC_SERIES]
    series = [
        [getattr(p.metrics, attr) for p in ready]
        for _, attr, _ in METRIC_SERIES
    ]
    colors = [color for _, _, color in METRIC_SERIES]

    x = range(len(labels))
    width = 0.24
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, name, values, color in zip(offsets, metric_names, series, colors):
        bars = ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=name,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_xlabel("Model")
    ax.set_title(title)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved plot to: {output_path}")


def print_table(points: list[ModelPoint]) -> None:
    print(
        f"{'Layers':>6}  {'Params':>12}  {'Acc':>6}  {'BalBal':>6}  "
        f"{'TPR':>6}  {'TNR':>6}  {'Status':>8}  Source"
    )
    print("-" * 105)
    for point in points:
        m = point.metrics
        fmt = lambda v: f"{v:.1f}" if v is not None else "—"
        status = "pending" if m.pending or m.accuracy is None else "ready"
        params = f"{point.num_params:,}"
        print(
            f"{point.layers:>4}L  {params:>12}  {fmt(m.accuracy):>6}  "
            f"{fmt(m.bal_bal_acc):>6}  {fmt(m.tpr):>6}  {fmt(m.tnr):>6}  "
            f"{status:>8}  {m.source}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=FIGURES_DIR / "bert_scaling_curve_v6.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Plot with interim / non-v6 numbers (default: only v6 mode=all).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=FIGURES_DIR / "bert_metrics_bars_v6.png",
        help="Output PNG for bal-bal / TPR / TNR grouped bar chart.",
    )
    parser.add_argument(
        "--log-plot-output",
        type=Path,
        default=FIGURES_DIR / "bert_combined_logscale_v6.png",
        help="Output PNG for BERT + LLM broken-axis log-scale plot (option 3).",
    )
    parser.add_argument(
        "--heatmap-output",
        type=Path,
        default=FIGURES_DIR / "bert_combined_heatmap_v6.png",
        help="Output PNG for BERT + LLM broken-axis heatmap.",
    )
    parser.add_argument(
        "--detection-fa-output",
        type=Path,
        default=FIGURES_DIR / "bert_detection_false_alarm_v6.png",
        help="Output PNG for detection recall vs false alarm grouped bars (paper fig 2).",
    )
    parser.add_argument(
        "--llm-suite",
        default=None,
        help=(
            "If set, use this suite for all LLMs under runs/llm_baseline/. "
            "Default: Gemma p3, Qwen/Llama p8 (per-model map)."
        ),
    )
    parser.add_argument(
        "--only-log-plot",
        action="store_true",
        help="Generate only the combined BERT+LLM log-scale plot.",
    )
    parser.add_argument(
        "--only-heatmap",
        action="store_true",
        help="Generate only the combined BERT+LLM heatmap.",
    )
    parser.add_argument(
        "--only-detection-fa",
        action="store_true",
        help="Generate only the detection recall vs false alarm bar chart.",
    )
    parser.add_argument(
        "--skip-log-plot",
        action="store_true",
        help="Skip the combined BERT+LLM log-scale plot.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip the combined BERT+LLM heatmap.",
    )
    parser.add_argument(
        "--skip-detection-fa",
        action="store_true",
        help="Skip the detection recall vs false alarm bar chart.",
    )
    parser.add_argument(
        "--all-plots",
        action="store_true",
        help="Generate log plot, heatmap, and legacy BERT-only plots.",
    )
    parser.add_argument(
        "--with-heatmap",
        action="store_true",
        help="Also generate the BERT+LLM heatmap (off by default).",
    )
    parser.add_argument(
        "--with-legacy",
        action="store_true",
        help="Also generate legacy BERT-only scaling curve and bar chart.",
    )
    parser.add_argument(
        "--skip-scaling",
        action="store_true",
        help="Skip the legacy BERT-only scaling curve.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip the legacy BERT-only bar chart.",
    )
    parser.add_argument(
        "--title",
        default="BERT inconsistency detector (hard, cls600) — v6 eval",
    )
    parser.add_argument(
        "--metrics-title",
        default="BERT inconsistency detector (hard, cls600) — v6 eval",
    )
    parser.add_argument(
        "--log-plot-title",
        default="Inconsistency detection — v6 eval",
    )
    parser.add_argument(
        "--heatmap-title",
        default="Inconsistency detection — v6 eval",
    )
    parser.add_argument(
        "--detection-fa-title",
        default="Detection recall vs false alarm rate",
    )
    parser.add_argument(
        "--skip-ladder",
        action="store_true",
        help="Skip clinical-specificity ladder plots (recall, bal-bal, FPR).",
    )
    parser.add_argument(
        "--only-ladder",
        action="store_true",
        help="Generate only the clinical-specificity ladder plots.",
    )
    parser.add_argument(
        "--ladder-metric",
        choices=("recall", "bal_bal", "fpr", "both", "all"),
        default="all",
        help="Which ladder plots to generate when --skip-ladder is not set.",
    )
    parser.add_argument(
        "--show-individual-llms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay individual LLM lines on clinical-specificity ladder plots (default: on).",
    )
    args = parser.parse_args()

    if args.all_plots:
        skip_scaling = args.skip_scaling
        skip_metrics = args.skip_metrics
        skip_log = args.skip_log_plot
        skip_heatmap = args.skip_heatmap
        skip_detection_fa = args.skip_detection_fa
        skip_ladder = args.skip_ladder
    elif args.only_log_plot:
        skip_scaling = skip_metrics = skip_heatmap = skip_detection_fa = skip_ladder = True
        skip_log = False
    elif args.only_heatmap:
        skip_scaling = skip_metrics = skip_log = skip_detection_fa = skip_ladder = True
        skip_heatmap = False
    elif args.only_detection_fa:
        skip_scaling = skip_metrics = skip_log = skip_heatmap = skip_ladder = True
        skip_detection_fa = False
    elif args.only_ladder:
        skip_scaling = skip_metrics = skip_log = skip_heatmap = skip_detection_fa = True
        skip_ladder = False
    else:
        # Default: log-scale + detection/FPR bars + clinical-specificity ladders.
        skip_scaling = args.skip_scaling or not args.with_legacy
        skip_metrics = args.skip_metrics or not args.with_legacy
        skip_log = args.skip_log_plot
        skip_heatmap = not args.with_heatmap
        skip_detection_fa = args.skip_detection_fa
        skip_ladder = args.skip_ladder

    require_v6_all = not args.allow_partial
    points = collect_points(DEFAULT_RUNS, require_v6_all=require_v6_all)
    print_table(points)

    plottable = [
        p
        for p in points
        if p.metrics.bal_bal_acc is not None
        and p.metrics.tpr is not None
        and p.metrics.tnr is not None
        and (args.allow_partial or not p.metrics.pending)
    ]
    if not plottable:
        raise SystemExit(
            "No v6 (mode=all) eval summaries with bal-bal / TPR / TNR yet. "
            "Re-run with --allow-partial to preview interim numbers."
        )

    if not skip_scaling:
        plot_scaling_curve(
            plottable,
            points,
            output_path=args.output,
            title=args.title,
        )

    if not skip_metrics:
        plot_metrics_bars(
            plottable,
            output_path=args.metrics_output,
            title=args.metrics_title,
        )

    bert_columns = bert_plot_columns(plottable)
    llm_columns = load_llm_columns(DEFAULT_LLMS, llm_suite=args.llm_suite)

    if not skip_log:
        plot_combined_log_scale(
            bert_columns,
            llm_columns,
            output_path=args.log_plot_output,
            title=args.log_plot_title,
        )

    if not skip_detection_fa:
        plot_detection_vs_false_alarm(
            bert_columns,
            llm_columns,
            output_path=args.detection_fa_output,
            title=args.detection_fa_title,
            llm_suite_note=_llm_suite_footnote(args.llm_suite),
        )

    if not skip_heatmap:
        plot_combined_heatmap(
            bert_columns,
            llm_columns,
            output_path=args.heatmap_output,
            title=args.heatmap_title,
        )

    if not skip_ladder:
        if args.ladder_metric == "both":
            ladder_metrics: tuple[str, ...] = ("recall", "bal_bal")
        elif args.ladder_metric == "all":
            ladder_metrics = ladder_plot.ALL_LADDER_METRICS
        else:
            ladder_metrics = (args.ladder_metric,)
        ladder_plot.generate_ladder_plots(
            llm_suite=args.llm_suite,
            metrics=ladder_metrics,
            show_individual_llms=args.show_individual_llms,
        )


if __name__ == "__main__":
    main()
