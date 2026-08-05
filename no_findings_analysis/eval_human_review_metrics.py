#!/usr/bin/env python3
"""
Evaluate BERT no-findings labels (nf_label) against human review (human_nf).

Rows marked human_nf=x (indeterminate / inconsistent) are excluded from metrics.

Usage
-----
  conda run -n PTCT3 python eval_human_review_metrics.py
  conda run -n PTCT3 python eval_human_review_metrics.py --csv path/to/review.csv
  conda run -n PTCT3 python eval_human_review_metrics.py --output nf_human_review_cm.png
  conda run -n PTCT3 python eval_human_review_metrics.py --roc-output nf_human_review_roc.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = _DIR / "no_findings_test_set_human_review.csv"
DEFAULT_CM_PATH = _DIR / "nf_human_review_confusion_matrix.png"
DEFAULT_ROC_PATH = _DIR / "nf_human_review_roc.png"
DEFAULT_BERT_THRESHOLD = 0.80


def _parse_label(val: str) -> int | None:
    v = str(val).strip().lower()
    if v in ("0", "1"):
        return int(v)
    return None


def _parse_float(val: str, default: float = float("nan")) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float | int]:
    """Positive class = 1 (no findings)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(y_true)
    acc = (tp + tn) / n if n else 0.0
    prec_nf = tp / (tp + fp) if (tp + fp) else 0.0
    rec_nf = tp / (tp + fn) if (tp + fn) else 0.0
    f1_nf = 2 * prec_nf * rec_nf / (prec_nf + rec_nf) if (prec_nf + rec_nf) else 0.0
    prec_findings = tn / (tn + fn) if (tn + fn) else 0.0
    rec_findings = tn / (tn + fp) if (tn + fp) else 0.0
    f1_findings = (
        2 * prec_findings * rec_findings / (prec_findings + rec_findings)
        if (prec_findings + rec_findings)
        else 0.0
    )
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision_nf": prec_nf,
        "recall_nf": rec_nf,
        "f1_nf": f1_nf,
        "precision_findings": prec_findings,
        "recall_findings": rec_findings,
        "f1_findings": f1_findings,
    }


def load_eval_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))
    eval_rows: List[Dict[str, str]] = []
    for row in rows:
        human = str(row.get("human_nf", "")).strip().lower()
        if human == "x":
            continue
        human_i = _parse_label(human)
        bert_i = _parse_label(str(row.get("nf_label", "")))
        if human_i is None or bert_i is None:
            continue
        eval_rows.append(row)
    return eval_rows, rows


def summarize_nf_human_review(
    csv_path: Path | str = DEFAULT_CSV,
) -> Dict[str, float | int] | None:
    """Return nf_label vs human_nf summary for eval tables; None if unavailable."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        return None
    eval_rows, _ = load_eval_rows(csv_path)
    if not eval_rows:
        return None
    y_true = [_parse_label(r["human_nf"]) for r in eval_rows]
    y_pred = [_parse_label(r["nf_label"]) for r in eval_rows]
    y_score = [_parse_float(r.get("bert_sim", "")) for r in eval_rows]
    assert all(v is not None for v in y_true)
    assert all(v is not None for v in y_pred)
    if any(np.isnan(s) for s in y_score):
        return None
    m = binary_metrics(y_true, y_pred)
    try:
        roc = roc_metrics(y_true, y_score)
        auroc = float(roc["auroc"])
    except ValueError:
        auroc = float("nan")
    return {
        "n": int(m["n"]),
        "accuracy": float(m["accuracy"]),
        "auroc": auroc,
        "tp": int(m["tp"]),
        "tn": int(m["tn"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
    }


def plot_confusion_matrix(
    metrics: Dict[str, float | int],
    output_path: Path,
    *,
    n_excluded_x: int,
) -> None:
    """Rows = human label, cols = BERT nf_label."""
    matrix = [
        [int(metrics["tn"]), int(metrics["fp"])],
        [int(metrics["fn"]), int(metrics["tp"])],
    ]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["BERT: Findings (0)", "BERT: No findings (1)"])
    ax.set_yticks([0, 1], labels=["Human: Findings (0)", "Human: No findings (1)"])
    acc = metrics["accuracy"]
    title = (
        f"No-findings human review vs BERT (n={metrics['n']}, "
        f"excluded x={n_excluded_x})\n"
        f"Accuracy = {acc:.1%}"
    )
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            val = matrix[i][j]
            color = "white" if val > max(v for row in matrix for v in row) * 0.55 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=14)
    ax.set_xlabel("BERT prediction (nf_label)")
    ax.set_ylabel("Human label (human_nf)")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def roc_metrics(
    y_true: List[int],
    y_score: List[float],
) -> Dict[str, float | np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("AUROC requires both human_nf=0 and human_nf=1 in eval set")
    fpr, tpr, thresholds = roc_curve(y, scores, pos_label=1)
    auroc = float(roc_auc_score(y, scores))
    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def plot_roc_curve_with_threshold(
    y_true: List[int],
    y_score: List[float],
    roc: Dict[str, float | np.ndarray],
    output_path: Path,
    *,
    n_eval: int,
    n_excluded_x: int,
    bert_threshold: float,
) -> None:
    fpr = roc["fpr"]
    tpr = roc["tpr"]
    auroc = roc["auroc"]
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pred_at_thresh = (scores >= bert_threshold).astype(int)
    tp = int(np.sum((y == 1) & (pred_at_thresh == 1)))
    fp = int(np.sum((y == 0) & (pred_at_thresh == 1)))
    fn = int(np.sum((y == 1) & (pred_at_thresh == 0)))
    tn = int(np.sum((y == 0) & (pred_at_thresh == 0)))
    op_tpr = tp / (tp + fn) if (tp + fn) else 0.0
    op_fpr = fp / (fp + tn) if (fp + tn) else 0.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(fpr, tpr, color="#2b6cb0", linewidth=2, label=f"bert_sim (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="Random")
    ax.scatter(
        [op_fpr],
        [op_tpr],
        color="#c45c5c",
        s=80,
        zorder=5,
        label=f"Threshold {bert_threshold:.2f} (TPR={op_tpr:.3f}, FPR={op_fpr:.3f})",
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate (human findings, BERT score high)")
    ax.set_ylabel("True positive rate (human NF, BERT score high)")
    ax.set_title(
        f"ROC: bert_sim vs human_nf (n={n_eval}, excluded x={n_excluded_x})\n"
        f"AUROC = {auroc:.4f}"
    )
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def print_metrics(
    csv_path: Path,
    *,
    cm_path: Path | None = None,
    roc_path: Path | None = None,
    bert_threshold: float = DEFAULT_BERT_THRESHOLD,
) -> Dict[str, float | int]:
    eval_rows, all_rows = load_eval_rows(csv_path)
    y_true = [_parse_label(r["human_nf"]) for r in eval_rows]
    y_pred = [_parse_label(r["nf_label"]) for r in eval_rows]
    y_score = [_parse_float(r.get("bert_sim", "")) for r in eval_rows]
    assert all(v is not None for v in y_true)
    assert all(v is not None for v in y_pred)
    if any(np.isnan(s) for s in y_score):
        raise SystemExit("Some eval rows are missing bert_sim scores")

    m = binary_metrics(y_true, y_pred)
    roc = roc_metrics(y_true, y_score)
    n_x = sum(1 for r in all_rows if str(r.get("human_nf", "")).strip().lower() == "x")
    n_unlabeled = sum(
        1
        for r in all_rows
        if _parse_label(str(r.get("human_nf", ""))) is None
        and str(r.get("human_nf", "")).strip().lower() != "x"
    )

    print(f"Human review CSV: {csv_path}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Excluded (human_nf=x): {n_x}")
    if n_unlabeled:
        print(f"Excluded (unlabeled human_nf): {n_unlabeled}")
    print(f"Evaluated: {m['n']}")
    print()
    print("BERT nf_label vs human_nf (positive class = no findings / 1)")
    print("-" * 56)
    print(f"  Accuracy:           {m['accuracy']:.4f}  ({m['tp'] + m['tn']}/{m['n']})")
    print(f"  Precision (NF=1):   {m['precision_nf']:.4f}")
    print(f"  Recall (NF=1):      {m['recall_nf']:.4f}")
    print(f"  F1 (NF=1):          {m['f1_nf']:.4f}")
    print(f"  Precision (findings=0): {m['precision_findings']:.4f}")
    print(f"  Recall (findings=0):    {m['recall_findings']:.4f}")
    print(f"  F1 (findings=0):        {m['f1_findings']:.4f}")
    print()
    print("BERT bert_sim vs human_nf (continuous score, positive = NF / 1)")
    print("-" * 56)
    print(f"  AUROC:                {roc['auroc']:.4f}")
    print(f"  Score column:         bert_sim")
    print(f"  Threshold (nf_label): {bert_threshold:.2f}")
    print()
    print("Confusion matrix (rows=human, cols=BERT)")
    print("                 BERT=0   BERT=1")
    print(f"  human=0 (findings)  {m['tn']:5d}    {m['fp']:5d}")
    print(f"  human=1 (NF)        {m['fn']:5d}    {m['tp']:5d}")
    print()
    print("Error breakdown:")
    print(f"  False positive NF (BERT=1, human=0): {m['fp']}")
    print(f"  False negative NF (BERT=0, human=1): {m['fn']}")

    if cm_path is not None:
        plot_confusion_matrix(m, cm_path, n_excluded_x=n_x)
        print()
        print(f"Confusion matrix plot saved to: {cm_path}")

    if roc_path is not None:
        plot_roc_curve_with_threshold(
            y_true,
            y_score,
            roc,
            roc_path,
            n_eval=int(m["n"]),
            n_excluded_x=n_x,
            bert_threshold=bert_threshold,
        )
        print(f"ROC plot saved to: {roc_path}")

    return m


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERT no-findings accuracy on human-reviewed test subset"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CM_PATH,
        help="Path for confusion matrix PNG (use --no-plot to skip)",
    )
    parser.add_argument(
        "--roc-output",
        type=Path,
        default=DEFAULT_ROC_PATH,
        help="Path for ROC curve PNG (use --no-plot to skip)",
    )
    parser.add_argument(
        "--bert-threshold",
        type=float,
        default=DEFAULT_BERT_THRESHOLD,
        help="bert_sim threshold used for nf_label (marked on ROC plot)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip all plots")
    args = parser.parse_args()
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    print_metrics(
        args.csv,
        cm_path=None if args.no_plot else args.output,
        roc_path=None if args.no_plot else args.roc_output,
        bert_threshold=args.bert_threshold,
    )


if __name__ == "__main__":
    main()
