#!/usr/bin/env python3
"""
label_no_findings.py
====================
Identifies "no-findings" (fully normal PET/CT scan) conclusions across the
train / val / test splits using two complementary signals:

  Signal A – Regex: _is_no_findings() from bert_inconsistency_detector
             (kept as a diagnostic; NOT used for the final label)
  Signal B – BERT CLS cosine-similarity to canonical NF reference embeddings
             (conclusion-only encoding: [CLS] {conclusion} [SEP])

Final label: Signal B alone (BERT threshold-tunable via --threshold).
             A1 is shown for comparison but excluded from the final label
             because it has false positives (partial-NF conclusions) that
             BERT correctly rejects.

Outputs
-------
  no_findings_labels.csv     – all records from all splits with signal scores
  no_findings_review.csv     – patient_id / accession_number / study_date /
                               source_row / split / conclusion / bert_sim /
                               nf_label — for manual review / threshold tuning
  nf_bert_distribution.png   – BERT similarity distribution (tune threshold here)
  nf_signal_comparison.png   – bar chart: NF count per signal
  nf_bert_adds.txt           – unique conclusions BERT catches, regex misses
  nf_a1_bert_disagrees.txt   – unique conclusions A1 catches but BERT rejects
  nf_heuristic_extras.txt    – short conclusions missed by both A1 and BERT

Usage
-----
  conda run -n PTCT3 python label_no_findings.py
  conda run -n PTCT3 python label_no_findings.py --threshold 0.82
  conda run -n PTCT3 python label_no_findings.py --model-path ./bert-contrastive_bs128_090526
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from report_split_io import ReportRecord, load_report_records  # noqa: E402
from src.bert_inconsistency_detector import (  # noqa: E402
    TOKENIZER_PATH,
    TRAIN_FILE,
    VAL_FILE,
    TEST_FILE,
    MODEL_PATH as DEFAULT_MODEL_PATH,
    _is_no_findings,
    init_tokenizer,
)
from transformers import BertModel  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical NF reference conclusions (used to compute the BERT anchor embedding)
# ---------------------------------------------------------------------------
CANONICAL_NF_REFS: List[str] = [
    # Core variants
    "No FDG avid disease demonstrated.",
    "No evidence of FDG-avid disease.",
    "No FDG avid residual/recurrent disease demonstrated.",
    "No FDG avid disease.",
    "No FDG-avid disease.",
    "No FDG avid recurrent disease demonstrated.",
    "No FDG-avid recurrence demonstrated.",
    "No FDG avid malignancy identified.",
    # Additional references covering common missed phrasings
    "Normal study. No FDG avid malignancy demonstrated.",
    "No evidence of FDG avid residual or recurrent disease.",
    "No PET-CT evidence of disease.",
    "No PET CT evidence of disease.",
    "No PET-CT evidence of malignancy.",
    "There is no evidence of FDG avid disease.",
    "No avid local, nodal or distant disease demonstrated.",
    "No metabolic evidence of recurrent disease.",
    "No abnormal FDG uptake demonstrated.",
    "No FDG avid recurrence or residual disease demonstrated.",
    "No FDG avid relapse.",
    "No FDG avid malignancy.",
    "No FDG avid abnormality demonstrated.",
    "No FDG avid systemic disease."
]

BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_splits() -> List[tuple]:
    """Return list of (split_name, ReportRecord) for train + val + test."""
    entries = []
    for split, path in [("train", TRAIN_FILE), ("val", VAL_FILE), ("test", TEST_FILE)]:
        try:
            records = load_report_records(path)
            entries.extend((split, r) for r in records)
            print(f"  {split:5s}: {len(records):6,} records  ({path})")
        except FileNotFoundError:
            print(f"  {split:5s}: file not found ({path}) — skipped")
    return entries


def _encode_cls(texts: List[str], model: BertModel) -> torch.Tensor:
    """Return L2-normalised CLS embeddings for raw conclusion strings."""
    from src.bert_inconsistency_detector import tokenizer as _tok

    vecs: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            wrapped = [f"{_tok.cls_token} {t} {_tok.sep_token}" for t in batch]
            enc = _tok(
                wrapped,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                add_special_tokens=False,
            )
            enc = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :]
            vecs.append(F.normalize(cls, dim=-1).cpu())
    return torch.cat(vecs, dim=0)


def _dedup_max(pairs, key_fn, sim_fn, higher_is_better=True):
    """Deduplicate by key; keep entry with highest (or lowest) sim."""
    seen: dict = {}
    for item in pairs:
        k, s = key_fn(item), sim_fn(item)
        if k not in seen:
            seen[k] = (s, item)
        else:
            prev_s = seen[k][0]
            if (higher_is_better and s > prev_s) or (not higher_is_better and s < prev_s):
                seen[k] = (s, item)
    return [item for _, item in seen.values()]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_path: str, bert_sim_threshold: float, max_words: int,
        max_conclusion_words: int, output_dir: Path):

    print(f"Loading tokenizer …")
    init_tokenizer(TOKENIZER_PATH)

    print("Loading data splits …")
    entries = _load_all_splits()          # [(split, ReportRecord), …]
    if not entries:
        raise RuntimeError("No data loaded — check that the data files exist.")

    splits      = [s  for s, _ in entries]
    records     = [r  for _, r in entries]
    conclusions = [r.conclusion for r in records]
    n           = len(records)
    print(f"Total: {n:,} records")

    # --- Signal A: strict regex ---
    sig_a = [_is_no_findings(c) for c in conclusions]

    # --- Signal B: BERT CLS similarity ---
    print(f"\nLoading BERT model from {model_path} …")
    bert = BertModel.from_pretrained(model_path, add_pooling_layer=False).to(DEVICE)

    print("Encoding canonical NF references …")
    ref_vecs = _encode_cls(CANONICAL_NF_REFS, bert)
    # Normalise each reference vector individually — we take max similarity
    # across all references, not similarity to the mean.  This directly tests
    # "is this conclusion close to at least one canonical NF phrase?" which is
    # the right criterion.  The mean-embedding approach would instead test
    # proximity to the centroid, which drifts as references become more diverse.
    ref_vecs_norm = F.normalize(ref_vecs, dim=-1)  # [R, D]

    print(f"Encoding {n:,} conclusions …")
    conc_vecs = _encode_cls(conclusions, bert)
    conc_vecs_norm = F.normalize(conc_vecs, dim=-1)  # [N, D]

    # sim_matrix[i, j] = cosine similarity of conclusion i to reference j
    sim_matrix = (conc_vecs_norm @ ref_vecs_norm.T).numpy()  # [N, R]
    bert_sims      = sim_matrix.max(axis=1)   # best-matching reference per conclusion
    best_ref_idx   = sim_matrix.argmax(axis=1)
    sig_b = bert_sims >= bert_sim_threshold

    # --- Signal C: length heuristic (diagnostic only) ---
    def _heuristic(c: str) -> bool:
        c_l = c.lower().strip()
        return len(c_l.split()) <= max_words and "no" in c_l.split() and "fdg" in c_l
    sig_c = [_heuristic(c) for c in conclusions]

    # Word-count filter: conclusions longer than max_conclusion_words are excluded
    # from the NF pool even if BERT is confident. Long conclusions typically contain
    # patient-specific qualifications (sites, measurements) that disqualify them
    # as generic "no findings" conclusions.
    word_counts = np.array([len(c.split()) for c in conclusions])
    sig_short = word_counts <= max_conclusion_words

    # Final label: BERT AND short enough
    final_nf = sig_b & sig_short

    # ---------------------------------------------------------------------------
    # Write no_findings_labels.csv (full detail, all records)
    # ---------------------------------------------------------------------------
    labels_csv = output_dir / "no_findings_labels.csv"
    with open(labels_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id", "accession_number", "study_date", "source_row", "split",
            "conclusion",
            "word_count",
            "sig_a_regex",          # diagnostic: original strict regex
            "bert_sim",             # max cosine sim across all canonical refs
            "best_ref",             # which canonical phrase matched best
            "sig_b_bert",           # BERT at threshold (before word-count filter)
            "sig_c_heuristic",      # diagnostic: short-conclusion heuristic
            "nf_label",             # FINAL: BERT AND word_count <= max_conclusion_words
        ])
        for rec, sp, wc, a, sim, ref_i, b, heur, nf in zip(
            records, splits, word_counts, sig_a, bert_sims, best_ref_idx, sig_b, sig_c, final_nf
        ):
            writer.writerow([
                rec.patient_id, rec.accession_number, rec.study_date, rec.source_row, sp,
                rec.conclusion,
                int(wc), int(a), f"{sim:.4f}", CANONICAL_NF_REFS[int(ref_i)], int(b), int(heur), int(nf),
            ])
    print(f"\nSaved: {labels_csv}")

    # ---------------------------------------------------------------------------
    # Write no_findings_review.csv (slim, for manual review / threshold tuning)
    # ---------------------------------------------------------------------------
    review_csv = output_dir / "no_findings_review.csv"
    with open(review_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "patient_id", "accession_number", "study_date", "source_row", "split",
            "conclusion", "word_count", "bert_sim", "best_ref", "nf_label",
        ])
        for rec, sp, wc, sim, ref_i, nf in zip(records, splits, word_counts, bert_sims, best_ref_idx, final_nf):
            writer.writerow([
                rec.patient_id, rec.accession_number, rec.study_date, rec.source_row, sp,
                rec.conclusion, int(wc), f"{sim:.4f}", CANONICAL_NF_REFS[int(ref_i)], int(nf),
            ])
    print(f"Saved: {review_csv}")

    # ---------------------------------------------------------------------------
    # Summary — recompute everything from bert_sims (numpy array) to avoid any
    # stale-variable issues with boolean masks derived earlier in the pipeline.
    # ---------------------------------------------------------------------------
    _b_mask  = bert_sims >= bert_sim_threshold     # recompute from raw sims
    _sh_mask = word_counts <= max_conclusion_words
    _fn_mask = _b_mask & _sh_mask

    nf_a      = sum(sig_a)
    nf_b      = int(_b_mask.sum())
    nf_short  = int(_sh_mask.sum())
    nf_final  = int(_fn_mask.sum())
    nf_c      = sum(sig_c)
    nf_b_cut  = int((_b_mask & ~_sh_mask).sum())
    a1_only   = sum(a and not b for a, b in zip(sig_a, _b_mask))
    b_only    = sum(b and not a for a, b in zip(_b_mask, sig_a))
    a1_b      = sum(a and b     for a, b in zip(sig_a, _b_mask))

    # Sanity check: nf_final must match nf_label column written to CSV
    import csv as _csv
    _csv_nf = sum(1 for row in open(labels_csv) if row.split(',')[-1].strip() == '1')
    assert _csv_nf == nf_final, f"Stats/CSV mismatch: {_csv_nf} vs {nf_final}"

    print(f"\n{'='*70}")
    print(f"{'Signal':<50} {'NF':>7} {'%':>6}")
    print(f"{'-'*70}")
    print(f"{'A1: Regex v1 (diagnostic)':<50} {nf_a:>7,} {100*nf_a/n:>5.1f}%")
    print(f"{'B:  BERT CLS sim ≥ {:.2f}'.format(bert_sim_threshold):<50} {nf_b:>7,} {100*nf_b/n:>5.1f}%")
    print(f"{'    ≤{} words (word-count filter)'.format(max_conclusion_words):<50} {nf_short:>7,} {100*nf_short/n:>5.1f}%")
    print(f"{'FINAL: BERT AND ≤{} words  ← nf_label'.format(max_conclusion_words):<50} {nf_final:>7,} {100*nf_final/n:>5.1f}%")
    print(f"{'C:  Length heuristic ≤{} words (diagnostic)'.format(max_words):<50} {nf_c:>7,} {100*nf_c/n:>5.1f}%")
    print(f"{'='*70}")
    print(f"  BERT=1 excluded by word-count filter (>{max_conclusion_words}w): {nf_b_cut:,}")
    print(f"  A1 AND B (both agree — high confidence NF):  {a1_b:,}")
    print(f"  B only   (BERT adds, regex misses):          {b_only:,}")
    print(f"  A1 only  (regex catches, BERT rejects):      {a1_only:,}  ← not in final label")

    # ---------------------------------------------------------------------------
    # Diagnostic text files
    # ---------------------------------------------------------------------------

    # 1. Conclusions BERT adds that regex misses (useful for extending regex)
    bert_adds = [(conclusions[i], float(bert_sims[i]))
                 for i in range(n) if sig_b[i] and not sig_a[i]]
    bert_adds = _dedup_max(bert_adds, key_fn=lambda x: x[0],
                           sim_fn=lambda x: x[1], higher_is_better=True)
    bert_adds.sort(key=lambda x: -x[1])
    out = output_dir / "nf_bert_adds.txt"
    with open(out, "w") as f:
        f.write(f"BERT (sim ≥ {bert_sim_threshold}) catches but regex misses "
                f"({len(bert_adds)} unique):\n\n")
        for c, s in bert_adds:
            f.write(f"[{s:.4f}]  {c}\n")
    print(f"\nBERT adds {len(bert_adds)} unique conclusions not in regex → {out}")
    print("Top 20:")
    for c, s in bert_adds[:20]:
        print(f"  [{s:.4f}]  {c[:100]}")

    # 2. Conclusions A1 catches but BERT rejects (regex false positives)
    a1_disagrees = [(conclusions[i], float(bert_sims[i]))
                    for i in range(n) if sig_a[i] and not sig_b[i]]
    a1_disagrees = _dedup_max(a1_disagrees, key_fn=lambda x: x[0],
                              sim_fn=lambda x: x[1], higher_is_better=False)
    a1_disagrees.sort(key=lambda x: x[1])  # lowest BERT sim first = most suspicious
    out = output_dir / "nf_a1_bert_disagrees.txt"
    with open(out, "w") as f:
        f.write(f"A1 regex catches but BERT sim < {bert_sim_threshold} "
                f"({len(a1_disagrees)} unique — likely false positives):\n\n")
        for c, s in a1_disagrees:
            f.write(f"[{s:.4f}]  {c}\n")
    print(f"\nA1 catches but BERT disagrees — {len(a1_disagrees)} unique → {out}")
    print("Most suspicious (lowest BERT sim):")
    for c, s in a1_disagrees[:20]:
        print(f"  [{s:.4f}]  {c[:100]}")

    # 3. Short conclusions heuristic catches but neither A1 nor B does
    heur_extras = [(conclusions[i], float(bert_sims[i]), len(conclusions[i].split()))
                   for i in range(n) if sig_c[i] and not sig_a[i] and not sig_b[i]]
    seen_h: dict = {}
    for c, s, w in heur_extras:
        if c not in seen_h:
            seen_h[c] = (s, w)
    heur_extras_u = sorted(seen_h.items(), key=lambda x: (x[1][1], -x[1][0]))
    out = output_dir / "nf_heuristic_extras.txt"
    with open(out, "w") as f:
        f.write(f"Short conclusions (≤{max_words}w + 'no' + 'fdg') missed by A1 AND B "
                f"({len(heur_extras_u)} unique):\n")
        f.write(f"If these look genuinely NF, lower --threshold or add a regex pattern.\n\n")
        for c, (s, w) in heur_extras_u:
            f.write(f"[{s:.4f} | {w:2d}w]  {c}\n")
    print(f"\nHeuristic extras (missed by A1 and B): {len(heur_extras_u)} unique → {out}")
    print(f"[bert_sim | words]  conclusion  (showing first 10)")
    for c, (s, w) in heur_extras_u[:10]:
        print(f"  [{s:.4f} | {w:2d}w]  {c[:100]}")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------
    # 1. Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_labels = [
        "A1: Regex v1\n(diagnostic)",
        f"B: BERT ≥{bert_sim_threshold:.2f}\n(before length filter)",
        f"Final: BERT +\n≤{max_conclusion_words}w cutoff ✓",
        f"C: Heuristic\n(≤{max_words}w, diagnostic)",
    ]
    counts = [nf_a, nf_b, nf_final, nf_c]
    colors = ["#aec7e8", "#f7b47e", "#dd8452", "#c44e52"]
    bars = ax.bar(bar_labels, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                f"{cnt:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("# Conclusions labelled NF")
    ax.set_title(f"No-findings counts by signal  (N={n:,})\n"
                 f"Final label = BERT ≥{bert_sim_threshold:.2f} AND ≤{max_conclusion_words} words")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output_dir / "nf_signal_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")

    # 2. BERT similarity distribution
    #
    # HOW TO READ THIS PLOT / TUNE THE THRESHOLD
    # -------------------------------------------
    # X-axis: max cosine similarity across all canonical NF reference embeddings.
    # Orange = conclusions the strict regex labels NF (approximate ground truth).
    #          These should peak near 1.0 if BERT is well-calibrated for this task.
    # Blue   = all other conclusions (most are disease reports).
    #          Any blue bars right of the threshold line are genuine NF conclusions
    #          the regex missed — they go into nf_bert_adds.txt.
    # Red dashed line = current threshold.
    # Good threshold: sits in the VALLEY between the two peaks.
    # Move RIGHT → fewer, purer NF labels.  Move LEFT → more NF but risk
    # including disease conclusions.
    sim_a  = bert_sims[np.array(sig_a, dtype=bool)]
    sim_na = bert_sims[~np.array(sig_a, dtype=bool)]
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.linspace(-0.3, 1.0, 66)
    ax.hist(sim_na, bins=bins, alpha=0.55, color="#4c72b0",
            label="Regex = False (disease / unknown)")
    ax.hist(sim_a,  bins=bins, alpha=0.55, color="#dd8452",
            label="Regex = True  (confirmed NF)")
    ax.axvline(bert_sim_threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold {bert_sim_threshold:.2f}  →  {nf_b:,} labelled NF")
    ymax = ax.get_ylim()[1]
    ax.text(bert_sim_threshold + 0.01, ymax * 0.93,
            f"← NF by BERT\n   ({nf_b:,})", fontsize=8, color="red", va="top")
    ax.text(bert_sim_threshold - 0.01, ymax * 0.93,
            "NOT NF →", fontsize=8, color="gray", va="top", ha="right")
    ax.set_xlabel("BERT CLS max cosine similarity to any canonical NF reference")
    ax.set_ylabel("Count")
    ax.set_title("BERT NF similarity — set threshold at the valley between the two peaks")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = output_dir / "nf_bert_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nDone.")
    print(f"\nTo retune threshold without re-encoding, use the CSV:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{labels_csv}')")
    print(f"  for t in [0.75, 0.78, 0.80, 0.82, 0.85]:")
    print(f"      print(f'threshold={{t:.2f}}  NF={{( df.bert_sim >= t).sum()}}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label no-findings conclusions.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH,
                        help="BERT model for CLS encoding (default: %(default)s)")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="BERT cosine-similarity threshold (default: %(default)s)")
    parser.add_argument("--max-words", type=int, default=12,
                        help="Max words for length-heuristic diagnostic (default: %(default)s)")
    parser.add_argument("--max-conclusion-words", type=int, default=15,
                        help="Hard cutoff: conclusions longer than this are excluded from "
                             "the final NF label even if BERT is confident (default: %(default)s)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory (default: current directory)")
    args = parser.parse_args()

    _out = Path(args.output_dir)
    _out.mkdir(parents=True, exist_ok=True)

    run(
        model_path=args.model_path,
        bert_sim_threshold=args.threshold,
        max_words=args.max_words,
        max_conclusion_words=args.max_conclusion_words,
        output_dir=_out,
    )
