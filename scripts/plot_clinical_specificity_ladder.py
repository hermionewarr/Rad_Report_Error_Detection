#!/usr/bin/env python3
"""Plot detection recall, bal-bal accuracy, or FPR vs clinical specificity (BERT vs LLM)."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[1]
BERT_ROOT = REPO_ROOT
RUNS_DIR = BERT_ROOT / "trained_models" / "runs" / "bert_binary"
LLM_BASELINE_DIR = BERT_ROOT / "trained_models" / "runs" / "llm_baseline"
FIGURES_DIR = BERT_ROOT / "figures"

sys.path.insert(0, str(BERT_ROOT))
from data.build_eval_datasets import (  # noqa: E402
    LATERALITY_SWAP_TERMS,
    SITE_SWAP_RULES,
    SEVERITY_FLIP_TERMS,
    _is_no_findings,
    count_swap_keyword_incidence,
)

DEFAULT_BERT_RUNS: tuple[tuple[str, str], ...] = (
    ("BERT 6", "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-6"),
    ("BERT 8", "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-8L"),
    ("BERT 12", "20260629_bert_inconsistency-hard-single-mlmcls-preclshead-cls600-12L"),
)
DEFAULT_LLM_SUITE = "20260626_llm-suite__p3"
# Best prompt per model: Gemma p3; Qwen/Llama p8 (split across dated suite dirs).
DEFAULT_LLM_SUITES: dict[str, str] = {
    "gemma-3-27b-gguf": "20260626_llm-suite__p3",
    "qwen3-32b-gguf": "20260629_llm-suite__p8",
    "llama-3.3-70b": "20260630_llm-suite__p8",
}
DEFAULT_LLMS: tuple[tuple[str, str], ...] = (
    ("Gemma3-27B", "gemma-3-27b-gguf"),
    ("Qwen3-32B", "qwen3-32b-gguf"),
    ("Llama3-70B", "llama-3.3-70b"),
)
BERT_PALETTE = ["#08306B", "#4C72B0", "#9ECAE1"]
LLM_PALETTE = {
    "Llama3-70B": "#CCB974",
    "Qwen3-32B": "#8172B3",
    "Gemma3-27B": "#DA8BC3",
}
LLM_LEGEND_ORDER = ("Gemma3-27B", "Qwen3-32B", "Llama3-70B")

# Natural-language spatial descriptors vs medically specialised region terms.
SITE_NATLANG_TERMS = frozenset({"upper", "lower", "top", "bottom"})
SITE_REGION_TERMS = frozenset(
    {
        "apical",
        "basal",
        "proximal",
        "distal",
        "anterior",
        "posterior",
        "lung",
        "liver",
        "bone",
        "pleural",
        "parenchymal",
        "thoracic",
        "abdominal",
    }
)

SPECIFICITY_BINS: tuple[tuple[str, str], ...] = (
    ("cross_pair", "Conclusion swap"),
    ("nf_v_dis", "No find. ↔ disease"),
    ("laterality", "Laterality"),
    ("site", "Site"),
    ("severity", "Severity"),
)
BIN_KEYS = [key for key, _ in SPECIFICITY_BINS]

METRIC_SPECS: dict[str, dict[str, object]] = {
    "recall": {
        "ylabel": "Detection recall (%)",
        "csv_col": "recall_pct",
        "default_output": FIGURES_DIR / "clinical_specificity_ladder_v6.png",
        "default_csv": FIGURES_DIR / "clinical_specificity_ladder_v6.csv",
        "ylim": (45, 102),
        "n_label_y_axes": 0.03,
        "legend_y_axes": 0.13,
    },
    "bal_bal": {
        "ylabel": "Balanced-balanced accuracy (%)",
        "csv_col": "bal_bal_pct",
        "default_output": FIGURES_DIR / "clinical_specificity_ladder_balbal_v6.png",
        "default_csv": FIGURES_DIR / "clinical_specificity_ladder_balbal_v6.csv",
        "ylim": (60, 102),
        "n_label_y_axes": 0.03,
        "legend_y_axes": 0.13,
    },
    "fpr": {
        "ylabel": "False positive rate (%)",
        "csv_col": "fpr_pct",
        "default_output": FIGURES_DIR / "clinical_specificity_ladder_fpr_v6.png",
        "default_csv": FIGURES_DIR / "clinical_specificity_ladder_fpr_v6.csv",
        "ylim": (0, 42),
        "n_label_y_axes": 0.97,
        "legend_y_axes": 0.13,
        "lower_is_better": True,
    },
    "tpr": {
        "ylabel": "True positive rate (%)",
        "csv_col": "recall_pct",
        "default_output": FIGURES_DIR / "clinical_specificity_ladder_tpr_v6.png",
        "default_csv": FIGURES_DIR / "clinical_specificity_ladder_tpr_v6.csv",
        "ylim": (45, 102),
        "n_label_y_axes": 0.03,
        "legend_y_axes": 0.13,
    },
}

ALL_LADDER_METRICS = ("recall", "bal_bal", "fpr")
DEFAULT_TPR_FPR_OUTPUT = FIGURES_DIR / "clinical_specificity_ladder_tpr_fpr_v6.png"
DEFAULT_TPR_FPR_CSV = FIGURES_DIR / "clinical_specificity_ladder_tpr_fpr_v6.csv"


@dataclass
class BinStats:
    n: int = 0
    detected: int = 0

    @property
    def recall_pct(self) -> float:
        if self.n == 0:
            return float("nan")
        return 100.0 * self.detected / self.n


@dataclass
class ConsistentBinStats:
    n: int = 0
    false_alarms: int = 0

    @property
    def fpr_pct(self) -> float:
        if self.n == 0:
            return float("nan")
        return 100.0 * self.false_alarms / self.n


@dataclass
class ModelMetrics:
    consistent_recall_pct: float
    bins: Dict[str, BinStats]
    consistent_bins: Dict[str, ConsistentBinStats]

    def bin_value_pct(self, bin_key: str, metric: str) -> float:
        entry = self.bins[bin_key]
        if metric in ("recall", "tpr"):
            return entry.recall_pct
        if metric == "bal_bal":
            if entry.n == 0:
                return float("nan")
            return 0.5 * entry.recall_pct + 0.5 * self.consistent_recall_pct
        if metric == "fpr":
            return self.consistent_bins[bin_key].fpr_pct
        raise ValueError(f"Unknown metric: {metric}")


def _site_rule_groups() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    natlang: List[Tuple[str, str]] = []
    region: List[Tuple[str, str]] = []
    for left, right, _ in SITE_SWAP_RULES:
        pair = tuple(sorted((left.lower(), right.lower())))
        if left.lower() in SITE_NATLANG_TERMS and right.lower() in SITE_NATLANG_TERMS:
            natlang.append(pair)  # type: ignore[arg-type]
        elif left.lower() in SITE_REGION_TERMS and right.lower() in SITE_REGION_TERMS:
            region.append(pair)  # type: ignore[arg-type]
    return natlang, region


SITE_NATLANG_PAIRS, SITE_REGION_PAIRS = _site_rule_groups()


def _terms_from_pairs(pairs: Iterable[Tuple[str, str]]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for left, right in pairs:
        for term in (left, right):
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


SITE_NATLANG_TERM_LIST = _terms_from_pairs(SITE_NATLANG_PAIRS)
SITE_REGION_TERM_LIST = _terms_from_pairs(SITE_REGION_PAIRS)
DISEASE_CONCLUSION_RE = re.compile(r"\b(?:disease|recurrence|residual)\b", re.IGNORECASE)


def _swap_hit(findings: str, conclusion: str, left: str, right: str) -> bool:
    f = findings.lower()
    c = conclusion.lower()
    return (left in f and right in c) or (right in f and left in c)


def _match_rule_pairs(
    findings: str,
    conclusion: str,
    pairs: Iterable[Tuple[str, str]],
) -> bool:
    return any(_swap_hit(findings, conclusion, left, right) for left, right in pairs)


def classify_site_swap(findings: str, conclusion: str) -> str:
    del findings, conclusion  # all site_swap rows share one bin
    return "site"


def _consistent_matches_bin(bin_key: str, conclusion: str) -> bool:
    """Whether a consistent report belongs to a specificity bin's control profile."""
    lat_inc = count_swap_keyword_incidence(conclusion, LATERALITY_SWAP_TERMS)
    site_nat_inc = count_swap_keyword_incidence(conclusion, SITE_NATLANG_TERM_LIST)
    site_reg_inc = count_swap_keyword_incidence(conclusion, SITE_REGION_TERM_LIST)
    sev_inc = count_swap_keyword_incidence(conclusion, SEVERITY_FLIP_TERMS)

    if bin_key == "cross_pair":
        return lat_inc == 0 and site_nat_inc == 0 and site_reg_inc == 0 and sev_inc == 0
    if bin_key == "nf_v_dis":
        return _is_no_findings(conclusion) or bool(DISEASE_CONCLUSION_RE.search(conclusion))
    if bin_key == "laterality":
        return lat_inc >= 1
    if bin_key == "site":
        return site_nat_inc >= 1 or site_reg_inc >= 1
    if bin_key == "severity":
        return sev_inc >= 1
    return False


def specificity_bin(row: dict) -> Optional[str]:
    error_type = (row.get("error_type") or "").strip()
    if error_type == "consistent":
        return None
    if error_type == "cross_pair_mismatch":
        return "cross_pair"
    if error_type == "laterality_swap":
        return "laterality"
    if error_type == "site_swap":
        return classify_site_swap(row.get("findings", ""), row.get("conclusion", ""))
    if error_type == "severity_flip":
        return "severity"
    if error_type == "NF_v_Dis_swap":
        return "nf_v_dis"
    return None


def _row_detected(row: dict) -> bool:
    return (row.get("is_tp0") or "").strip() == "1"


def _keep_row(row: dict, *, require_parse_ok: bool) -> bool:
    if require_parse_ok and "parse_ok" in row:
        return (row.get("parse_ok") or "").strip() == "1"
    return True


def _row_consistent_correct(row: dict) -> bool:
    return (row.get("is_tn0") or "").strip() == "1"


def _row_false_alarm(row: dict) -> bool:
    return (row.get("is_fp0") or "").strip() == "1"


def load_model_metrics(
    predictions_csv: Path,
    *,
    require_parse_ok: bool = False,
) -> ModelMetrics:
    bins: Dict[str, BinStats] = {key: BinStats() for key in BIN_KEYS}
    consistent_bins: Dict[str, ConsistentBinStats] = {
        key: ConsistentBinStats() for key in BIN_KEYS
    }
    consistent_n = 0
    consistent_correct = 0

    with predictions_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not _keep_row(row, require_parse_ok=require_parse_ok):
                continue
            if (row.get("error_type") or "").strip() == "consistent":
                consistent_n += 1
                if _row_consistent_correct(row):
                    consistent_correct += 1
                conclusion = row.get("conclusion", "")
                is_fp = _row_false_alarm(row)
                for bin_key in BIN_KEYS:
                    if _consistent_matches_bin(bin_key, conclusion):
                        consistent_bins[bin_key].n += 1
                        if is_fp:
                            consistent_bins[bin_key].false_alarms += 1
                continue
            bin_key = specificity_bin(row)
            if bin_key is None:
                continue
            bins[bin_key].n += 1
            if _row_detected(row):
                bins[bin_key].detected += 1

    consistent_recall_pct = (
        100.0 * consistent_correct / consistent_n if consistent_n else float("nan")
    )
    return ModelMetrics(
        consistent_recall_pct=consistent_recall_pct,
        bins=bins,
        consistent_bins=consistent_bins,
    )


def load_bin_stats(predictions_csv: Path, *, require_parse_ok: bool = False) -> Dict[str, BinStats]:
    return load_model_metrics(
        predictions_csv,
        require_parse_ok=require_parse_ok,
    ).bins


def write_summary_csv(
    output_csv: Path,
    model_metrics: Dict[str, ModelMetrics],
    *,
    metric: str,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_col = str(METRIC_SPECS[metric]["csv_col"])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "specificity_bin",
                "label",
                "model",
                "n",
                "detected",
                "error_recall_pct",
                "consistent_n",
                "consistent_fp",
                "consistent_fpr_pct",
                "consistent_recall_pct",
                csv_col,
            ]
        )
        for bin_key, label in SPECIFICITY_BINS:
            for model_name, metrics in model_metrics.items():
                entry = metrics.bins[bin_key]
                con_entry = metrics.consistent_bins[bin_key]
                writer.writerow(
                    [
                        bin_key,
                        label,
                        model_name,
                        entry.n,
                        entry.detected,
                        f"{entry.recall_pct:.2f}" if entry.n else "",
                        con_entry.n,
                        con_entry.false_alarms,
                        f"{con_entry.fpr_pct:.2f}" if con_entry.n else "",
                        f"{metrics.consistent_recall_pct:.2f}",
                        f"{metrics.bin_value_pct(bin_key, metric):.2f}"
                        if entry.n or (metric == "fpr" and con_entry.n)
                        else "",
                    ]
                )
    print(f"Saved summary CSV to: {output_csv}")


def _ordered_legend(
    bert_names: Sequence[str],
    *,
    show_individual_llms: bool,
) -> tuple[list[object], list[str]]:
    """Legend: BERT 6→12, BERT range, Gemma→Qwen→Llama, LLM range."""
    handles: list[object] = []
    labels: list[str] = []

    for idx, bert_name in enumerate(bert_names):
        handles.append(
            Line2D(
                [0],
                [0],
                color=BERT_PALETTE[idx % len(BERT_PALETTE)],
                marker="o",
                linewidth=2.0,
                markersize=7,
            )
        )
        labels.append(bert_name)

    if len(bert_names) > 1:
        handles.append(Patch(facecolor="#4C72B0", alpha=0.30, edgecolor="none"))
        labels.append("BERT range")

    if show_individual_llms:
        for llm_name in LLM_LEGEND_ORDER:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=LLM_PALETTE[llm_name],
                    marker=".",
                    linewidth=1.2,
                    markersize=6,
                )
            )
            labels.append(llm_name)

    handles.append(Patch(facecolor="#DD8452", alpha=0.20, edgecolor="none"))
    labels.append("LLM range")

    return handles, labels


def _add_side_legend(
    fig: plt.Figure,
    handles: Sequence[object],
    labels: Sequence[str],
    *,
    plot_rect: tuple[float, float, float, float] = (0.0, 0.0, 0.85, 1.0),
) -> None:
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(plot_rect[2] - 0.02, 0.5),
        ncol=1,
        frameon=True,
        fontsize=7.5,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        borderpad=0.7,
        labelspacing=0.9,
        handletextpad=0.45,
    )
    fig.tight_layout(rect=plot_rect)


def _draw_ladder_on_axes(
    ax: plt.Axes,
    model_metrics: Dict[str, ModelMetrics],
    *,
    bert_names: Sequence[str],
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    n_label_y_axes: float,
    show_individual_llms: bool,
    show_n_labels: bool = True,
    show_gap_annotations: bool = False,
) -> None:
    lower_is_better = bool(METRIC_SPECS[metric].get("lower_is_better"))
    x = np.arange(len(SPECIFICITY_BINS))
    x_labels = [label for _, label in SPECIFICITY_BINS]

    llm_names = [name for name in model_metrics if name not in bert_names]

    def _values(name: str) -> list[float]:
        return [model_metrics[name].bin_value_pct(key, metric) for key in BIN_KEYS]

    llm_matrix = np.array([_values(name) for name in llm_names])
    llm_mean = np.nanmean(llm_matrix, axis=0)
    llm_min = np.nanmin(llm_matrix, axis=0)
    llm_max = np.nanmax(llm_matrix, axis=0)

    bert_matrix = np.array([_values(name) for name in bert_names])
    bert_min = np.nanmin(bert_matrix, axis=0)
    bert_max = np.nanmax(bert_matrix, axis=0)

    if lower_is_better:
        all_vals = np.concatenate([bert_matrix.ravel(), llm_matrix.ravel()])
        finite = all_vals[np.isfinite(all_vals)]
        ymax = min(100.0, float(np.ceil(max(finite.max() + 4, 12))))
        ylim = (0.0, ymax)

    ax.fill_between(
        x,
        llm_min,
        llm_max,
        color="#DD8452",
        alpha=0.20,
        zorder=1,
    )
    if len(bert_names) > 1:
        ax.fill_between(
            x,
            bert_min,
            bert_max,
            color="#4C72B0",
            alpha=0.30,
            zorder=2,
        )

    for idx, bert_name in enumerate(bert_names):
        ax.plot(
            x,
            _values(bert_name),
            color=BERT_PALETTE[idx % len(BERT_PALETTE)],
            linewidth=2.0 if idx == len(bert_names) - 1 else 1.8,
            marker="o",
            markersize=7,
            zorder=5 + idx,
        )

    if show_individual_llms:
        for llm_name in LLM_LEGEND_ORDER:
            if llm_name not in model_metrics:
                continue
            ax.plot(
                x,
                _values(llm_name),
                color=LLM_PALETTE[llm_name],
                linewidth=1.2,
                marker=".",
                markersize=6,
                alpha=0.85,
                zorder=3,
            )

    if show_n_labels:
        ref_stats = model_metrics[bert_names[-1]].bins
        ref_consistent = model_metrics[bert_names[-1]].consistent_bins
        xaxis_transform = ax.get_xaxis_transform()
        for xi, (key, _label) in enumerate(SPECIFICITY_BINS):
            if metric == "fpr":
                n = ref_consistent[key].n
            else:
                n = ref_stats[key].n
            ax.text(
                xi,
                n_label_y_axes,
                f"n={n}",
                transform=xaxis_transform,
                ha="center",
                va="bottom" if not lower_is_better else "top",
                fontsize=7.5,
                color="#555555",
                clip_on=False,
            )

    if show_gap_annotations and len(bert_names) == 1 and metric in ("recall", "tpr", "fpr"):
        for xi, (key, _label) in enumerate(SPECIFICITY_BINS):
            bert_val = model_metrics[bert_names[0]].bin_value_pct(key, metric)
            llm_val = llm_mean[xi]
            if not (np.isfinite(bert_val) and np.isfinite(llm_val)):
                continue
            if metric in ("recall", "tpr"):
                gap = bert_val - llm_val
                gap_ok = gap > 0.5
            else:
                gap = llm_val - bert_val
                gap_ok = gap > 0.5
            if gap_ok:
                mid_y = 0.5 * (bert_val + llm_val)
                ax.annotate(
                    f"−{gap:.0f} pp" if metric == "fpr" else f"+{gap:.0f} pp",
                    (xi, mid_y),
                    textcoords="offset points",
                    xytext=(8, 0),
                    ha="left",
                    va="center",
                    fontsize=8,
                    color="#666666",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=18, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Increasing clinical specificity →")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save figure as PNG (dpi=200) and a matching PDF beside it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved plot to: {pdf_path}")


def plot_ladder(
    model_metrics: Dict[str, ModelMetrics],
    *,
    bert_names: Sequence[str],
    output_path: Path,
    ylabel: str,
    ylim: tuple[float, float],
    n_label_y_axes: float,
    legend_y_axes: float,
    metric: str,
    show_individual_llms: bool,
) -> None:
    del legend_y_axes
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(10.0, 4.0))
    _draw_ladder_on_axes(
        ax,
        model_metrics,
        bert_names=bert_names,
        metric=metric,
        ylabel=ylabel,
        ylim=ylim,
        n_label_y_axes=n_label_y_axes,
        show_individual_llms=show_individual_llms,
        show_gap_annotations=True,
    )

    legend_handles, legend_labels = _ordered_legend(
        bert_names,
        show_individual_llms=show_individual_llms,
    )
    _add_side_legend(fig, legend_handles, legend_labels)

    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_tpr_fpr_panels(
    model_metrics: Dict[str, ModelMetrics],
    *,
    bert_names: Sequence[str],
    output_path: Path,
    show_individual_llms: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    tpr_spec = METRIC_SPECS["tpr"]
    fpr_spec = METRIC_SPECS["fpr"]
    #fig, (ax_tpr, ax_fpr) = plt.subplots(1, 2, figsize=(12.5, 4.0), sharex=True)
    fig, (ax_tpr, ax_fpr) = plt.subplots(1, 2, figsize=(14, 4.0), sharex=True)

    _draw_ladder_on_axes(
        ax_tpr,
        model_metrics,
        bert_names=bert_names,
        metric="tpr",
        ylabel=str(tpr_spec["ylabel"]),
        ylim=tpr_spec["ylim"],  # type: ignore[arg-type]
        n_label_y_axes=float(tpr_spec["n_label_y_axes"]),
        show_individual_llms=show_individual_llms,
    )

    _draw_ladder_on_axes(
        ax_fpr,
        model_metrics,
        bert_names=bert_names,
        metric="fpr",
        ylabel=str(fpr_spec["ylabel"]),
        ylim=fpr_spec["ylim"],  # type: ignore[arg-type]
        n_label_y_axes=float(fpr_spec["n_label_y_axes"]),
        show_individual_llms=show_individual_llms,
    )

    legend_handles, legend_labels = _ordered_legend(
        bert_names,
        show_individual_llms=show_individual_llms,
    )
    _add_side_legend(fig, legend_handles, legend_labels)

    _save_figure(fig, output_path)
    plt.close(fig)


def print_table(model_metrics: Dict[str, ModelMetrics], *, metric: str) -> None:
    header = f"{'Bin':<22}" + "".join(f"{name:>14}" for name in model_metrics)
    print(header)
    print("-" * len(header))
    for bin_key, label in SPECIFICITY_BINS:
        row = f"{label:<22}"
        for metrics in model_metrics.values():
            has_value = metrics.bins[bin_key].n or (
                metric == "fpr" and metrics.consistent_bins[bin_key].n
            )
            if has_value:
                row += f"{metrics.bin_value_pct(bin_key, metric):>12.1f}%  "
            else:
                row += f"{'—':>14}"
        print(row)


def _parse_bert_runs(raw: Optional[str]) -> tuple[tuple[str, str], ...]:
    if not raw:
        return DEFAULT_BERT_RUNS
    runs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            label, run_name = item.split("=", 1)
            runs.append((label.strip(), run_name.strip()))
        else:
            runs.append((Path(item).name, item))
    if not runs:
        raise SystemExit("No BERT runs parsed from --bert-runs")
    return tuple(runs)


def _resolve_llm_suite(slug: str, llm_suite: Optional[str] = None) -> str:
    if llm_suite:
        return llm_suite
    return DEFAULT_LLM_SUITES.get(slug, DEFAULT_LLM_SUITE)


def load_all_model_metrics(
    *,
    bert_runs: tuple[tuple[str, str], ...] = DEFAULT_BERT_RUNS,
    llm_suite: Optional[str] = None,
    llms: tuple[tuple[str, str], ...] = DEFAULT_LLMS,
    skip_missing_bert: bool = False,
) -> tuple[Dict[str, ModelMetrics], list[str]]:
    model_metrics: Dict[str, ModelMetrics] = {}
    bert_names: list[str] = []

    for bert_label, run_name in bert_runs:
        bert_csv = RUNS_DIR / run_name / "results" / "hard_predictions.csv"
        if not bert_csv.exists():
            if skip_missing_bert:
                print(f"Skipping missing BERT predictions: {bert_csv}")
                continue
            raise FileNotFoundError(f"Missing BERT predictions: {bert_csv}")
        model_metrics[bert_label] = load_model_metrics(bert_csv)
        bert_names.append(bert_label)

    if not bert_names:
        raise FileNotFoundError("No BERT prediction files found for the requested runs")

    for llm_label, slug in llms:
        suite = _resolve_llm_suite(slug, llm_suite)
        llm_csv = (
            LLM_BASELINE_DIR / suite / "models" / slug / "results" / "predictions.csv"
        )
        if not llm_csv.exists():
            raise FileNotFoundError(f"Missing LLM predictions: {llm_csv}")
        print(f"Loading {llm_label} from suite {suite}")
        model_metrics[llm_label] = load_model_metrics(llm_csv, require_parse_ok=True)

    return model_metrics, bert_names


def generate_ladder_plots(
    *,
    bert_runs: tuple[tuple[str, str], ...] = DEFAULT_BERT_RUNS,
    llm_suite: Optional[str] = None,
    metrics: Sequence[str] = ALL_LADDER_METRICS,
    show_individual_llms: bool = True,
    skip_missing_bert: bool = True,
    output: Optional[Path] = None,
    bal_bal_output: Optional[Path] = None,
    fpr_output: Optional[Path] = None,
    tpr_fpr_output: Optional[Path] = None,
    summary_csv: Optional[Path] = None,
    bal_bal_summary_csv: Optional[Path] = None,
    fpr_summary_csv: Optional[Path] = None,
    tpr_fpr_summary_csv: Optional[Path] = None,
) -> None:
    """Generate one or more clinical-specificity ladder plots (callable from other scripts)."""
    model_metrics, bert_names = load_all_model_metrics(
        bert_runs=bert_runs,
        llm_suite=llm_suite,
        skip_missing_bert=skip_missing_bert,
    )

    ladder_metrics = [metric for metric in metrics if metric != "tpr_fpr"]
    if "tpr_fpr" in metrics:
        print("\n=== tpr_fpr ===")
        print_table(model_metrics, metric="tpr")
        print_table(model_metrics, metric="fpr")
        tpr_fpr_csv = Path(tpr_fpr_summary_csv or DEFAULT_TPR_FPR_CSV)
        write_summary_csv(tpr_fpr_csv.with_name(tpr_fpr_csv.stem + "_tpr.csv"), model_metrics, metric="tpr")
        write_summary_csv(tpr_fpr_csv.with_name(tpr_fpr_csv.stem + "_fpr.csv"), model_metrics, metric="fpr")
        plot_tpr_fpr_panels(
            model_metrics,
            bert_names=bert_names,
            output_path=Path(tpr_fpr_output or DEFAULT_TPR_FPR_OUTPUT),
            show_individual_llms=show_individual_llms,
        )

    output_by_metric = {
        "recall": output,
        "bal_bal": bal_bal_output,
        "fpr": fpr_output,
    }
    summary_by_metric = {
        "recall": summary_csv,
        "bal_bal": bal_bal_summary_csv,
        "fpr": fpr_summary_csv,
    }

    for metric in ladder_metrics:
        if metric not in METRIC_SPECS:
            raise ValueError(f"Unknown metric: {metric}")
        spec = METRIC_SPECS[metric]
        print(f"\n=== {metric} ===")
        print_table(model_metrics, metric=metric)

        out_path = Path(output_by_metric[metric] or spec["default_output"])
        summary_path = Path(summary_by_metric[metric] or spec["default_csv"])

        write_summary_csv(summary_path, model_metrics, metric=metric)
        plot_ladder(
            model_metrics,
            bert_names=bert_names,
            output_path=out_path,
            ylabel=str(spec["ylabel"]),
            ylim=spec["ylim"],  # type: ignore[arg-type]
            n_label_y_axes=float(spec["n_label_y_axes"]),
            legend_y_axes=float(spec["legend_y_axes"]),
            metric=metric,
            show_individual_llms=show_individual_llms,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bert-runs",
        default="",
        help=(
            "Comma-separated BERT runs as LABEL=run_dir. "
            "Default: BERT 6, 8, and 12 cls600 hard-single runs."
        ),
    )
    parser.add_argument(
        "--llm-suite",
        default=None,
        help=(
            "If set, use this suite for all LLMs under runs/llm_baseline/. "
            "Default: Gemma from p3, Qwen/Llama from p8."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=("recall", "bal_bal", "fpr", "tpr_fpr", "both", "all"),
        default="tpr_fpr",
        help="Y-axis metric: recall, bal-bal, FPR, side-by-side TPR+FPR, or combinations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (recall plot when --metric all/both/recall).",
    )
    parser.add_argument(
        "--bal-bal-output",
        type=Path,
        default=None,
        help="Output PNG path for bal-bal plot (when --metric all/both/bal_bal).",
    )
    parser.add_argument(
        "--fpr-output",
        type=Path,
        default=None,
        help="Output PNG path for FPR plot (when --metric all/fpr).",
    )
    parser.add_argument(
        "--tpr-fpr-output",
        type=Path,
        default=None,
        help="Output PNG path for side-by-side TPR/FPR plot.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Summary CSV for recall plot.",
    )
    parser.add_argument(
        "--bal-bal-summary-csv",
        type=Path,
        default=None,
        help="Summary CSV for bal-bal plot.",
    )
    parser.add_argument(
        "--fpr-summary-csv",
        type=Path,
        default=None,
        help="Summary CSV for FPR plot.",
    )
    parser.add_argument(
        "--tpr-fpr-summary-csv",
        type=Path,
        default=None,
        help="Summary CSV base path for side-by-side TPR/FPR plot.",
    )
    parser.add_argument(
        "--skip-missing-bert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip BERT runs without predictions (default: on).",
    )
    parser.add_argument(
        "--show-individual-llms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay each LLM line in addition to the range (default: on).",
    )
    args = parser.parse_args()

    if args.metric == "both":
        metrics: tuple[str, ...] = ("recall", "bal_bal")
    elif args.metric == "all":
        metrics = (*ALL_LADDER_METRICS, "tpr_fpr")
    else:
        metrics = (args.metric,)

    generate_ladder_plots(
        bert_runs=_parse_bert_runs(args.bert_runs),
        llm_suite=args.llm_suite,
        metrics=metrics,
        show_individual_llms=args.show_individual_llms,
        skip_missing_bert=args.skip_missing_bert,
        output=args.output,
        bal_bal_output=args.bal_bal_output,
        fpr_output=args.fpr_output,
        tpr_fpr_output=args.tpr_fpr_output,
        summary_csv=args.summary_csv,
        bal_bal_summary_csv=args.bal_bal_summary_csv,
        fpr_summary_csv=args.fpr_summary_csv,
        tpr_fpr_summary_csv=args.tpr_fpr_summary_csv,
    )


if __name__ == "__main__":
    main()
