"""bert_inconsistency_multiclass.py – Multiclass error-type classifier.

Extends the binary inconsistency detector (bert_inconsistency_detector.py) to
predict the *type* of inconsistency rather than just consistent vs inconsistent.

Label classes (LABEL_MAP):
  0  consistent          – findings and conclusion are consistent  (error_type="positive" in CSVs)
  1  laterality_swap     – left/right swap in conclusion
  2  site_swap           – anatomical site swapped in conclusion
  3  severity_flip       – severity qualifier swapped in conclusion
  4  NF_v_Dis_swap       – no-findings vs disease conclusion swapped
  5  cross_pair_mismatch – generic mismatched conclusion (incl. random_mismatch)

All tokenisation, error generation, data loading, and pool utilities are
imported from bert_inconsistency_detector.py — nothing in that module is
modified.  The multiclass head replaces the single-logit BCE head with a
Linear(hidden, N_CLASSES) + CrossEntropyLoss head.
"""
import argparse
import csv
import json as _json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel
import wandb

try:
    import src.bert_inconsistency_detector as _bid
    from src.bert_inconsistency_detector import (
        BATCH_SIZE,
        DEVICE,
        DEFAULT_SEQ_LENGTH,
        EPOCHS,
        LR,
        NF_LABELS_CSV,
        SUPPORTED_SEQ_LENGTHS,
        SEED,
        TOKENIZER_PATH,
        TRAIN_FILE,
        VAL_FILE,
        HardNegativeTrainingDataset,
        InconsistencyDataset,
        PairExample,
        SWAP_INSTANCES_CHOICES,
        format_joint_report,
        init_tokenizer,
        load_nf_labels_from_csv,
        normalize_error_type_for_reporting,
        parse_findings_conclusion,
        set_seed,
        _metadata_from_eval_csv_row,
    )
except ImportError:
    import sys
    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    import src.bert_inconsistency_detector as _bid
    from src.bert_inconsistency_detector import (
        BATCH_SIZE,
        DEVICE,
        DEFAULT_SEQ_LENGTH,
        EPOCHS,
        LR,
        NF_LABELS_CSV,
        SUPPORTED_SEQ_LENGTHS,
        SEED,
        TOKENIZER_PATH,
        TRAIN_FILE,
        VAL_FILE,
        HardNegativeTrainingDataset,
        InconsistencyDataset,
        PairExample,
        SWAP_INSTANCES_CHOICES,
        format_joint_report,
        init_tokenizer,
        load_nf_labels_from_csv,
        normalize_error_type_for_reporting,
        parse_findings_conclusion,
        set_seed,
        _metadata_from_eval_csv_row,
    )

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
from paths import (  # noqa: E402
    checkpoint_path,
    current_argv,
    default_plot_path,
    default_result_path,
    setup_run,
)
from src.nf_swap_eval import (  # noqa: E402
    print_nf_swap_direction_table,
    summarize_nf_swap_by_direction,
)

DEFAULT_CHECKPOINT = "bert-inconsistency-multiclass"
SCRIPT_ID = "bert-inconsistency-mc"
RUN_CATEGORY = "bert_multiclass"


def default_model_dir() -> str:
    return checkpoint_path(DEFAULT_CHECKPOINT)

MC_CLS_HEAD_FILE = "mc_classifier.pt"

# ─────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────
LABEL_MAP: Dict[str, int] = {
    "positive": 0,             # legacy CSV name for consistent pairs
    "consistent": 0,
    "laterality_swap": 1,
    "site_swap": 2,
    "severity_flip": 3,
    "NF_v_Dis_swap": 4,
    "cross_pair_mismatch": 5,
    "random_mismatch": 5,      # alias — same error family (original train_mode)
}


def _normalize_error_type_for_label(error_type: str) -> str:
    """Map legacy/alternate names before LABEL_MAP lookup."""
    error_type = normalize_error_type_for_reporting(error_type)
    if error_type == "positive":
        return "consistent"
    return error_type


def label_from_error_type(error_type: str) -> int:
    """Multiclass index for an error_type string."""
    return LABEL_MAP.get(
        _normalize_error_type_for_label(error_type),
        LABEL_MAP["cross_pair_mismatch"],
    )


# Display names (index → human-readable label shown in metrics and plots)
LABEL_NAMES: Dict[int, str] = {
    0: "consistent",
    1: "laterality_swap",
    2: "site_swap",
    3: "severity_flip",
    4: "NF_v_Dis_swap",
    5: "cross_pair_mismatch",
}
N_CLASSES = 6

# Upweights rule-based corruptions that often fail on first draw (see error_type_analysis.ipynb).
# Passed to HardNegativeTrainingDataset.error_type_weights when --balanced_error_weights is set.
BALANCED_ERROR_TYPE_WEIGHTS: Dict[str, float] = {
    "laterality_swap": 3.0,
    "site_swap": 3.0,
    "severity_flip": 3.0,
    "NF_v_Dis_swap": 1.0,
    "cross_pair_mismatch": 1.0,
}
HARD_GENERATOR_ERROR_TYPES = frozenset(BALANCED_ERROR_TYPE_WEIGHTS.keys())


def resolve_error_type_weights(
    error_type_weights_json: Optional[str],
    use_balanced_preset: bool,
) -> Optional[Dict[str, float]]:
    """Parse CLI weights; optional preset overrides uniform 1.0 sampling."""
    if error_type_weights_json:
        weights = _json.loads(error_type_weights_json)
        if not isinstance(weights, dict):
            raise ValueError("--error_type_weights must be a JSON object")
        unknown = set(weights) - HARD_GENERATOR_ERROR_TYPES
        if unknown:
            raise ValueError(
                f"Unknown error_type_weights keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(HARD_GENERATOR_ERROR_TYPES)}"
            )
        return {k: float(v) for k, v in weights.items()}
    if use_balanced_preset:
        return dict(BALANCED_ERROR_TYPE_WEIGHTS)
    return None


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class MulticlassConsistencyClassifier(nn.Module):
    """BERT backbone + Linear(hidden, N_CLASSES) head with CrossEntropyLoss."""

    def __init__(self, bert: BertModel, classifier_head: nn.Linear):
        super().__init__()
        self.bert = bert
        self.classifier = classifier_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
        )
        cls_repr = outputs.last_hidden_state[:, 0, :]   # [B, H]
        logits = self.classifier(cls_repr)               # [B, N_CLASSES]

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return type(
            "MulticlassOutput",
            (),
            {
                "logits": logits,
                "loss": loss,
                "attentions": getattr(outputs, "attentions", None),
            },
        )()


# ─────────────────────────────────────────────
# COLLATE FUNCTIONS
# (_bid.tokenizer is the live global set by init_tokenizer())
# ─────────────────────────────────────────────
def collate_fn_multiclass(batch: List[PairExample]):
    """Training collate: maps error_type → class index for multiclass labels."""
    texts = [format_joint_report(x.findings, x.conclusion) for x in batch]
    labels = torch.tensor(
        [label_from_error_type(x.error_type) for x in batch],
        dtype=torch.long,
    )
    encoded = _bid.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=_bid.MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded["labels"] = labels
    return encoded


def collate_eval_csv_multiclass(batch: List[PairExample]):
    """Eval collate: label already is class index (set by CsvEvalDatasetMulticlass)."""
    texts = [format_joint_report(x.findings, x.conclusion) for x in batch]
    labels = torch.tensor([x.label for x in batch], dtype=torch.long)
    error_types = [x.error_type for x in batch]

    encoded = _bid.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=_bid.MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded["labels"] = labels
    return encoded, error_types


# ─────────────────────────────────────────────
# EVAL DATASET
# ─────────────────────────────────────────────
class CsvEvalDatasetMulticlass(Dataset):
    """Reads a standard eval CSV (findings, conclusion, label, error_type) and
    derives the multiclass label from the error_type column.

    The binary `label` column (0/1) is ignored for training — the ground truth
    class comes from error_type via LABEL_MAP.  Unknown error_type strings fall
    back to class 5 (cross_pair_mismatch).
    """

    def __init__(self, csv_path: str, mode: str = "all"):
        self.examples: List[PairExample] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"findings", "conclusion", "label", "error_type"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

            for row in reader:
                findings = (row.get("findings") or "").strip()
                conclusion = (row.get("conclusion") or "").strip()
                error_type = _normalize_error_type_for_label(
                    (row.get("error_type") or "unknown").strip()
                )
                if not findings or not conclusion:
                    continue

                if mode == "cross_pair_full":
                    if error_type not in {"consistent", "cross_pair_mismatch"}:
                        continue

                mc_label = label_from_error_type(error_type)
                patient_id, source_idx = _metadata_from_eval_csv_row(row)
                self.examples.append(
                    PairExample(
                        findings,
                        conclusion,
                        mc_label,
                        error_type,
                        patient_id=patient_id,
                        source_idx=source_idx,
                    )
                )

        print(f"Loaded {len(self.examples)} eval rows from {csv_path} (mode={mode})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict:
    """Compute overall, binary-detection, and per-class multiclass metrics.

    Returns a dict with:
      acc              – exact multiclass accuracy
      binary_acc       – accuracy treating class 0 as consistent, 1+ as inconsistent
      binary_precision – precision of detecting any error (class != 0)
      binary_recall    – recall of detecting any error
      binary_f1        – F1 of detecting any error
      macro_f1         – unweighted mean F1 over classes with support > 0
      weighted_f1      – support-weighted mean F1
      per_class        – {class_idx: {precision, recall, f1, support, tp, fp, fn}}
    """
    preds = torch.argmax(logits, dim=1)
    acc = (preds == labels).float().mean().item()

    # Binary-detection view: consistent (0) vs any error (1+)
    binary_preds = (preds != 0).long()
    binary_labels = (labels != 0).long()
    binary_acc = (binary_preds == binary_labels).float().mean().item()

    tp_bin = ((binary_preds == 1) & (binary_labels == 1)).sum().item()
    fp_bin = ((binary_preds == 1) & (binary_labels == 0)).sum().item()
    fn_bin = ((binary_preds == 0) & (binary_labels == 1)).sum().item()
    bin_prec = tp_bin / (tp_bin + fp_bin + 1e-8)
    bin_rec  = tp_bin / (tp_bin + fn_bin + 1e-8)
    bin_f1   = 2 * bin_prec * bin_rec / (bin_prec + bin_rec + 1e-8)

    # Per-class metrics
    per_class: Dict[int, Dict] = {}
    for c in range(N_CLASSES):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        support = int((labels == c).sum().item())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    active = [c for c in range(N_CLASSES) if per_class[c]["support"] > 0]
    macro_f1 = sum(per_class[c]["f1"] for c in active) / max(len(active), 1)
    total_sup = sum(per_class[c]["support"] for c in active)
    weighted_f1 = (
        sum(per_class[c]["f1"] * per_class[c]["support"] for c in active)
        / max(total_sup, 1)
    )

    return {
        "acc": acc,
        "binary_acc": binary_acc,
        "binary_precision": bin_prec,
        "binary_recall": bin_rec,
        "binary_f1": bin_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


# ─────────────────────────────────────────────
# EVAL FUNCTIONS
# ─────────────────────────────────────────────
def run_eval_multiclass(model, dataloader) -> Dict:
    """Eval over training-style loader (collate_fn_multiclass, no error_types tuple)."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            all_logits.append(out.logits.cpu())
            all_labels.append(batch["labels"].cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = multiclass_metrics(logits, labels)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def run_eval_csv_multiclass(
    model,
    dataloader,
    examples: Optional[List[PairExample]] = None,
    eval_csv: Optional[str] = None,
) -> Dict:
    """Eval over CsvEvalDatasetMulticlass loader (returns batch, error_types tuples).

    Returns overall_metrics dict which includes:
      - Standard multiclass metrics (acc, macro_f1, weighted_f1, per_class)
      - Binary detection view (binary_acc, binary_f1)
      - errors_only_detection_acc: among error examples, fraction where model predicts non-zero
      - errors_only_type_acc:      among error examples, fraction where model predicts correct class
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    all_error_types: List[str] = []

    with torch.no_grad():
        for batch, error_types in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            total_loss += out.loss.item()
            all_logits.append(out.logits.cpu())
            all_labels.append(batch["labels"].cpu())
            all_error_types.extend(error_types)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds  = torch.argmax(logits, dim=1)

    overall = multiclass_metrics(logits, labels)
    overall["loss"] = total_loss / len(dataloader)

    # Errors-only view (excludes consistent/positive rows)
    non_pos_mask = torch.tensor(
        [_normalize_error_type_for_label(et) != "consistent" for et in all_error_types],
        dtype=torch.bool,
    )
    n_err = int(non_pos_mask.sum().item())
    if n_err > 0:
        err_preds  = preds[non_pos_mask]
        err_labels = labels[non_pos_mask]
        overall["errors_only_detection_acc"] = (err_preds != 0).float().mean().item()
        overall["errors_only_type_acc"]      = (err_preds == err_labels).float().mean().item()
        overall["errors_only_n"] = n_err
    else:
        overall["errors_only_detection_acc"] = float("nan")
        overall["errors_only_type_acc"]      = float("nan")
        overall["errors_only_n"] = 0

    if examples is not None and len(examples) == len(preds):
        overall["nf_swap_by_direction"] = summarize_nf_swap_by_direction(
            examples,
            preds.numpy(),
            labels.numpy(),
            task="multiclass",
            eval_csv=eval_csv,
            nf_multiclass_id=4,
        )

    return overall


def save_eval_predictions_csv_multiclass(
    model,
    dataset: CsvEvalDatasetMulticlass,
    output_csv_path: str,
):
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_eval_csv_multiclass,
    )

    rows = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            bsz = batch["labels"].shape[0]
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch_gpu)
            probs = torch.softmax(out.logits, dim=1).cpu()   # [B, N_CLASSES]
            preds = torch.argmax(probs, dim=1).cpu()
            labels = batch["labels"].cpu()

            for i in range(bsz):
                ex = dataset.examples[offset + i]
                pred_cls  = int(preds[i].item())
                true_cls  = int(labels[i].item())
                prob_vec  = probs[i].tolist()
                rows.append(
                    {
                        "patient_id": ex.patient_id,
                        "source_idx": ex.source_idx if ex.source_idx is not None else "",
                        "findings": ex.findings,
                        "conclusion": ex.conclusion,
                        "error_type": ex.error_type,
                        "true_label": true_cls,
                        "true_label_name": LABEL_NAMES.get(true_cls, "unknown"),
                        "pred_label": pred_cls,
                        "pred_label_name": LABEL_NAMES.get(pred_cls, "unknown"),
                        "is_correct": int(pred_cls == true_cls),
                        "is_correct_binary": int((pred_cls == 0) == (true_cls == 0)),
                        **{f"prob_{LABEL_NAMES[c]}": f"{prob_vec[c]:.6f}" for c in range(N_CLASSES)},
                    }
                )
            offset += bsz

    fieldnames = [
        "patient_id", "source_idx",
        "findings", "conclusion", "error_type",
        "true_label", "true_label_name",
        "pred_label", "pred_label_name",
        "is_correct", "is_correct_binary",
        *[f"prob_{LABEL_NAMES[c]}" for c in range(N_CLASSES)],
    ]
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved row-level predictions to: {output_csv_path}")


# ─────────────────────────────────────────────
# EVAL ENTRY POINT
# ─────────────────────────────────────────────
def _print_classification_report(overall: Dict) -> None:
    """Print a clean sklearn-style classification report from multiclass_metrics output."""
    sep = "─" * 62
    header = f"  {'class':>5}  {'label':<22}  {'n':>5}  {'P':>6}  {'R':>6}  {'F1':>6}"
    print(f"\n{sep}")
    print(header)
    print(sep)
    for c in range(N_CLASSES):
        m = overall["per_class"][c]
        print(
            f"  {c:>5}  {LABEL_NAMES[c]:<22}  {m['support']:>5}  "
            f"{m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}"
        )
    print(sep)
    print(f"  {'':>5}  {'macro avg':<22}  {'':>5}  {'':>6}  {'':>6}  {overall['macro_f1']:>6.4f}")
    print(f"  {'':>5}  {'weighted avg':<22}  {'':>5}  {'':>6}  {'':>6}  {overall['weighted_f1']:>6.4f}")
    print(sep)


def eval_from_csv_multiclass(
    eval_csv: str,
    model_dir: str,
    mode: str = "all",
    save_predictions_csv: Optional[str] = None,
) -> Dict:
    bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
    if bert.get_input_embeddings().weight.shape[0] != len(_bid.tokenizer):
        bert.resize_token_embeddings(len(_bid.tokenizer))
    classifier_head = nn.Linear(bert.config.hidden_size, N_CLASSES)
    head_path = Path(model_dir) / MC_CLS_HEAD_FILE
    if not head_path.exists():
        raise FileNotFoundError(f"Missing multiclass classifier head: {head_path}")
    classifier_head.load_state_dict(torch.load(head_path, map_location="cpu"))
    model = MulticlassConsistencyClassifier(bert, classifier_head).to(DEVICE)

    dataset = CsvEvalDatasetMulticlass(eval_csv, mode=mode)
    if len(dataset) == 0:
        raise ValueError(f"No rows available after applying mode='{mode}' to {eval_csv}")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_eval_csv_multiclass,
    )

    overall = run_eval_csv_multiclass(
        model, dataloader, examples=dataset.examples, eval_csv=eval_csv
    )

    print(
        f"\n[overall]  loss={overall['loss']:.4f}  acc={overall['acc']:.4f}  "
        f"macro_f1={overall['macro_f1']:.4f}  weighted_f1={overall['weighted_f1']:.4f}"
    )
    print(
        f"[binary detection]  acc={overall['binary_acc']:.4f}  "
        f"precision={overall['binary_precision']:.4f}  "
        f"recall={overall['binary_recall']:.4f}  "
        f"f1={overall['binary_f1']:.4f}"
    )
    print(
        f"[errors only]  n={overall['errors_only_n']}  "
        f"detection_acc={overall['errors_only_detection_acc']:.4f}  "
        f"type_acc={overall['errors_only_type_acc']:.4f}"
    )
    _print_classification_report(overall)
    if overall.get("nf_swap_by_direction"):
        print_nf_swap_direction_table(overall["nf_swap_by_direction"], task="multiclass")

    if save_predictions_csv:
        save_eval_predictions_csv_multiclass(model, dataset, save_predictions_csv)
        try:
            from src.predictions_eval import write_eval_artifacts
        except ImportError:
            from predictions_eval import write_eval_artifacts  # type: ignore
        run_name = Path(save_predictions_csv).resolve().parent.parent.name
        write_eval_artifacts(
            predictions_csv=save_predictions_csv,
            run_name=run_name,
            eval_mode=mode,
            eval_csv=eval_csv,
            source="bert",
            write_confusion=False,
        )

    return overall


def eval_benchmarks_multiclass(
    model_dir: str,
    eval_csv_hard: str,
    eval_csv_cross_pair: str,
    hard_predictions_csv: Optional[str] = None,
    cross_pair_predictions_csv: Optional[str] = None,
    hard_confusion_matrix_path: Optional[str] = None,
    cross_pair_confusion_matrix_path: Optional[str] = None,
) -> None:
    """Run hard and cross-pair benchmarks separately (mirrors binary eval_benchmarks)."""
    print("\n=== Hard Mismatch Benchmark (multiclass) ===")
    eval_from_csv_multiclass(
        eval_csv_hard,
        model_dir,
        mode="hard",
        save_predictions_csv=hard_predictions_csv,
    )
    if hard_confusion_matrix_path:
        bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
        if bert.get_input_embeddings().weight.shape[0] != len(_bid.tokenizer):
            bert.resize_token_embeddings(len(_bid.tokenizer))
        head = nn.Linear(bert.config.hidden_size, N_CLASSES)
        head.load_state_dict(
            torch.load(Path(model_dir) / MC_CLS_HEAD_FILE, map_location="cpu")
        )
        eval_model = MulticlassConsistencyClassifier(bert, head).to(DEVICE)
        ds = CsvEvalDatasetMulticlass(eval_csv_hard, mode="hard")
        plot_confusion_matrix_multiclass(eval_model, ds, hard_confusion_matrix_path)

    print("\n=== Cross-Pair Benchmark (multiclass) ===")
    eval_from_csv_multiclass(
        eval_csv_cross_pair,
        model_dir,
        mode="cross_pair_full",
        save_predictions_csv=cross_pair_predictions_csv,
    )
    if cross_pair_confusion_matrix_path:
        bert = BertModel.from_pretrained(model_dir, add_pooling_layer=False)
        if bert.get_input_embeddings().weight.shape[0] != len(_bid.tokenizer):
            bert.resize_token_embeddings(len(_bid.tokenizer))
        head = nn.Linear(bert.config.hidden_size, N_CLASSES)
        head.load_state_dict(
            torch.load(Path(model_dir) / MC_CLS_HEAD_FILE, map_location="cpu")
        )
        eval_model = MulticlassConsistencyClassifier(bert, head).to(DEVICE)
        ds = CsvEvalDatasetMulticlass(eval_csv_cross_pair, mode="cross_pair_full")
        plot_confusion_matrix_multiclass(eval_model, ds, cross_pair_confusion_matrix_path)


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix_multiclass(
    model,
    dataset: CsvEvalDatasetMulticlass,
    output_path: str,
):
    """Build and save a full N×N confusion matrix from the dataset."""
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_eval_csv_multiclass
    )

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch, _ in dataloader:
            batch_gpu = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch_gpu)
            all_preds.append(torch.argmax(out.logits, dim=1).cpu())
            all_labels.append(batch["labels"].cpu())

    preds  = torch.cat(all_preds,  dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    matrix = [[0] * N_CLASSES for _ in range(N_CLASSES)]
    for true, pred in zip(labels, preds):
        matrix[int(true)][int(pred)] += 1

    class_names = [LABEL_NAMES[c] for c in range(N_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Multiclass Confusion Matrix")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved confusion matrix to: {output_path}")


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_multiclass(
    train_mode: str = "hard",
    corruption_prob: float = 0.5,
    negatives_per_positive: int = 1,
    swap_instances: str = "single",
    nf_labels_csv: str = NF_LABELS_CSV,
    nf_swap_nf_to_disease_frac: Optional[float] = 0.5,
    val_hard_eval_csv: Optional[str] = None,
    grad_clip: float = 1.0,
    error_type_weights: Optional[Dict[str, float]] = None,
    severity_jitter_prob: float = 0.0,
    use_wandb: bool = False,
    wandb_project: str = "bert-inconsistency-mc",
    wandb_run_name: Optional[str] = None,
    output_model_dir: Optional[str] = None,
) -> str:
    if not output_model_dir:
        raise ValueError("output_model_dir is required for training")
    set_seed(SEED)

    train_pairs = parse_findings_conclusion(TRAIN_FILE)
    val_pairs   = parse_findings_conclusion(VAL_FILE)
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)}")

    nf_labels_train = load_nf_labels_from_csv(nf_labels_csv, split="train")

    if swap_instances not in SWAP_INSTANCES_CHOICES:
        raise ValueError(
            f"swap_instances must be one of {SWAP_INSTANCES_CHOICES}, got {swap_instances!r}"
        )

    if train_mode == "original":
        train_dataset = InconsistencyDataset(
            train_pairs, negatives_per_positive=negatives_per_positive
        )
    else:
        train_dataset = HardNegativeTrainingDataset(
            train_pairs,
            corruption_prob=corruption_prob,
            nf_labels=nf_labels_train,
            error_type_weights=error_type_weights,
            nf_swap_nf_to_disease_frac=nf_swap_nf_to_disease_frac,
            swap_instances=swap_instances,
            severity_jitter_prob=severity_jitter_prob,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn_multiclass,
    )

    # Val loader: prefer the pre-built hard eval CSV for a proper multiclass signal.
    if val_hard_eval_csv and Path(val_hard_eval_csv).exists():
        val_eval_dataset = CsvEvalDatasetMulticlass(val_hard_eval_csv, mode="hard")
        val_loader = DataLoader(
            val_eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_eval_csv_multiclass,
        )
        use_csv_val = True
        print(f"Val eval: hard CSV '{val_hard_eval_csv}' ({len(val_eval_dataset)} rows)")
    else:
        if val_hard_eval_csv:
            print(
                f"Warning: val_hard_eval_csv '{val_hard_eval_csv}' not found; "
                "falling back to random val negatives (class 0 vs class 5 only)."
            )
        val_dataset = InconsistencyDataset(val_pairs, negatives_per_positive=1)
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn_multiclass,
        )
        use_csv_val = False

    print(
        f"Training mode: {train_mode} | init_model: {_bid.MODEL_PATH} "
        f"| corruption_prob: {corruption_prob:.2f} "
        f"| swap_instances: {swap_instances} "
        f"| n_classes: {N_CLASSES}"
    )

    wandb_enabled = bool(use_wandb)
    if wandb_enabled:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "train_mode": train_mode,
                "train_model_path": _bid.MODEL_PATH,
                "corruption_prob": corruption_prob,
                "negatives_per_positive": negatives_per_positive,
                "swap_instances": swap_instances,
                "n_classes": N_CLASSES,
                "label_map": LABEL_MAP,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "max_length": _bid.MAX_LENGTH,
                "device": DEVICE,
                "grad_clip": grad_clip,
                "val_hard_eval_csv": val_hard_eval_csv or "random_negatives",
                "error_type_weights": error_type_weights or "uniform",
                "severity_jitter_prob": severity_jitter_prob,
            },
        )

    bert = BertModel.from_pretrained(_bid.MODEL_PATH, add_pooling_layer=False)
    bert.resize_token_embeddings(len(_bid.tokenizer))
    classifier_head = nn.Linear(bert.config.hidden_size, N_CLASSES)
    model = MulticlassConsistencyClassifier(bert, classifier_head).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            optimizer.zero_grad()
            out.loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += out.loss.item()

        avg_train_loss = total_loss / len(train_loader)

        if use_csv_val:
            metrics = run_eval_csv_multiclass(model, val_loader)
        else:
            metrics = run_eval_multiclass(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train_loss={avg_train_loss:.4f} "
            f"| val_loss={metrics['loss']:.4f} "
            f"| val_acc={metrics['acc']:.4f} "
            f"| val_binary_acc={metrics['binary_acc']:.4f} "
            f"| val_binary_f1={metrics['binary_f1']:.4f} "
            f"| val_macro_f1={metrics['macro_f1']:.4f}"
        )

        if wandb_enabled:
            log_dict: Dict = {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": metrics["loss"],
                "val/acc": metrics["acc"],
                "val/binary_acc": metrics["binary_acc"],
                "val/binary_precision": metrics["binary_precision"],
                "val/binary_recall": metrics["binary_recall"],
                "val/binary_f1": metrics["binary_f1"],
                "val/macro_f1": metrics["macro_f1"],
                "val/weighted_f1": metrics["weighted_f1"],
            }
            if use_csv_val:
                log_dict["val/errors_only_detection_acc"] = metrics.get("errors_only_detection_acc", float("nan"))
                log_dict["val/errors_only_type_acc"] = metrics.get("errors_only_type_acc", float("nan"))
            # Per-class F1 to track per error type over training
            for c in range(N_CLASSES):
                pc = metrics["per_class"][c]
                if pc["support"] > 0:
                    log_dict[f"val/f1_{LABEL_NAMES[c]}"] = pc["f1"]
            wandb.log(log_dict)

    out_dir = Path(output_model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bert.save_pretrained(out_dir)
    torch.save(classifier_head.state_dict(), out_dir / MC_CLS_HEAD_FILE)
    _bid.tokenizer.save_pretrained(out_dir)
    print(f"Saved multiclass inconsistency detector to: {out_dir}")
    if wandb_enabled:
        wandb.finish()
    return str(out_dir)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multiclass inconsistency detector — predicts error type."
    )

    # ── Eval ──────────────────────────────────
    parser.add_argument("--eval_csv", type=str, default=None,
        help="Skip training and evaluate this CSV (findings, conclusion, label, error_type).")
    parser.add_argument("--model_dir", type=str, default=None,
        help="Model path for --eval_csv / --eval_only (default: latest multiclass run).")
    parser.add_argument("--output_model_dir", type=str, default=None,
        help="Override model output directory for training.")
    parser.add_argument("--run_name", type=str, default=None,
        help="Optional run label; saved as YYYYMMDD_<name> under trained_models/runs/.")
    parser.add_argument("--mode", type=str, default="all",
        choices=["all", "hard", "cross_pair_full"],
        help=(
            "Eval row filter: all=no filter; hard=full hard-negative v4 suite "
            "(all error types in CSV, including cross_pair_mismatch); "
            "cross_pair_full=consistent+cross_pair_mismatch only."
        ),
    )
    parser.add_argument("--save_predictions_csv", type=str, default=None,
        help="Row-level predictions CSV path (single eval mode).")
    parser.add_argument("--confusion_matrix_path", type=str, default=None,
        help="Optional path to save the N×N confusion matrix plot.")
    parser.add_argument("--eval_csv_hard", type=str, default=None,
        help="Hard eval CSV; with --eval_csv_cross_pair runs dual benchmarks.")
    parser.add_argument("--eval_csv_cross_pair", type=str, default=None,
        help="Cross-pair eval CSV; with --eval_csv_hard runs dual benchmarks.")
    parser.add_argument("--hard_predictions_csv", type=str, default=None,
        help="Predictions CSV for hard benchmark in dual eval mode.")
    parser.add_argument("--cross_pair_predictions_csv", type=str, default=None,
        help="Predictions CSV for cross-pair benchmark in dual eval mode.")
    parser.add_argument("--hard_confusion_matrix_path", type=str, default=None,
        help="Confusion matrix PNG for hard benchmark in dual eval mode.")
    parser.add_argument("--cross_pair_confusion_matrix_path", type=str, default=None,
        help="Confusion matrix PNG for cross-pair benchmark in dual eval mode.")
    parser.add_argument("--eval_only", action="store_true",
        help="Skip training; run evaluation with --eval_csv.")

    # ── Training ──────────────────────────────
    parser.add_argument("--train_mode", type=str, default="hard",
        choices=["original", "hard"],
        help="original=random cross-pair negatives, hard=on-the-fly hard corruptions.")
    parser.add_argument("--corruption_prob", type=float, default=0.5)
    parser.add_argument(
        "--swap_instances",
        type=str,
        choices=SWAP_INSTANCES_CHOICES,
        default="single",
        help=(
            "Hard mode keyword swap scope for laterality/site/severity errors: "
            "all=swap every match; single=swap exactly one trigger word (random position); "
            "random=same as single; random_n=swap a random count in [1..N]. "
            "Default: single (matches binary hard training)."
        ),
    )
    parser.add_argument("--negatives_per_positive", type=int, default=1,
        help="Original mode only.")
    parser.add_argument("--nf_labels_csv", type=str, default=NF_LABELS_CSV)
    parser.add_argument(
        "--nf_swap_nf_to_disease_frac",
        type=float,
        default=0.5,
        help=(
            "Hard training only: when NF_v_Dis_swap is sampled, target this fraction "
            "of NF→disease swaps (re-sources NF-original reports when needed). "
            "Set to -1 to disable direction balancing (legacy behavior)."
        ),
    )
    parser.add_argument("--val_hard_eval_csv", type=str, default=None,
        help="Pre-built hard eval CSV for the val split (enables multiclass val metrics).")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--error_type_weights",
        type=str,
        default=None,
        help=(
            "Hard mode only: JSON dict of relative sampling weights for on-the-fly "
            "corruption generators (same as bert_inconsistency_detector.py). "
            "Example: '{\"laterality_swap\": 3.0, \"site_swap\": 3.5, "
            "\"severity_flip\": 6.0, \"NF_v_Dis_swap\": 1.0, "
            "\"cross_pair_mismatch\": 1.0}'. Unlisted types default to 1.0. "
            "Overrides --balanced_error_weights when both are set."
        ),
    )
    parser.add_argument(
        "--balanced_error_weights",
        action="store_true",
        help=(
            "Hard mode only: use built-in BALANCED_ERROR_TYPE_WEIGHTS to upweight "
            "laterality/site/severity draws (NF and cross_pair stay at 1.0)."
        ),
    )
    parser.add_argument(
        "--severity_jitter_prob",
        type=float,
        default=0.0,
        help=(
            "Hard mode only: on uncorrupted (consistent) training rows, probability of "
            "within-tier severity synonym jitter in the conclusion (artifact mitigation). "
            "Matches bert_inconsistency_detector.py / run_bert_sequence.sh."
        ),
    )

    # ── WandB ─────────────────────────────────
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="bert-inconsistency-mc")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    parser.add_argument(
        "--seq_length",
        type=int,
        choices=SUPPORTED_SEQ_LENGTHS,
        default=DEFAULT_SEQ_LENGTH,
        help=(
            "Max token sequence length for training/eval "
            f"(default: {DEFAULT_SEQ_LENGTH}). "
            "Init checkpoint is bert_cls_classification_mxln{seq_length}[_big|_8L|_12L] "
            "unless --wo_cls_init is set."
        ),
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="One of small/big/8L/12L (or 4/6/8/12). Used to resolve init checkpoint.",
    )
    parser.add_argument(
        "--wo_cls_init",
        action="store_true",
        help=(
            "Init from bert-pretrained_mxlen_{seq_length} (MLM pretrain only) instead "
            "of bert_cls_classification_mxln{seq_length}. Ignored when --init_checkpoint "
            "is set."
        ),
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help=(
            "Optional init checkpoint name (e.g. bert_cls_classification_mxln600_8L) "
            "or path to a model directory with config.json + weights. Overrides the "
            "default from --seq_length / --model_size / --wo_cls_init."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training/eval batch size (default: 128 for small/big/8L, 64 for 12L).",
    )
    args = parser.parse_args()
    try:
        from bert_architectures import default_cls_batch_size
    except ImportError:
        from bert_architectures import default_cls_batch_size  # type: ignore[no-redef]
    train_batch_size = args.batch_size or default_cls_batch_size(args.model_size)
    _bid.configure_seq_length(
        args.seq_length,
        wo_cls_init=args.wo_cls_init,
        init_checkpoint=args.init_checkpoint,
        model_size=args.model_size,
        batch_size=train_batch_size,
    )
    init_tokenizer(TOKENIZER_PATH)
    if args.eval_only:
        args.model_dir = args.model_dir or default_model_dir()

    def _run_tags() -> tuple[str, ...]:
        tags = [args.train_mode]
        if args.balanced_error_weights:
            tags.append("balanced")
        return tuple(tags)

    if args.eval_only:
        eval_layout = setup_run(
            SCRIPT_ID,
            tags=("eval", *_run_tags()),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            argv=current_argv(),
        )
        print(f"Run outputs: {eval_layout.root}")
        run_dual_eval = bool(args.eval_csv_hard and args.eval_csv_cross_pair)
        if run_dual_eval:
            eval_benchmarks_multiclass(
                model_dir=args.model_dir,
                eval_csv_hard=args.eval_csv_hard,
                eval_csv_cross_pair=args.eval_csv_cross_pair,
                hard_predictions_csv=default_result_path(
                    eval_layout, "hard_predictions.csv", args.hard_predictions_csv
                ),
                cross_pair_predictions_csv=default_result_path(
                    eval_layout, "cross_pair_predictions.csv", args.cross_pair_predictions_csv
                ),
                hard_confusion_matrix_path=default_plot_path(
                    eval_layout, "hard_confusion_matrix.png", args.hard_confusion_matrix_path
                ),
                cross_pair_confusion_matrix_path=default_plot_path(
                    eval_layout,
                    "cross_pair_confusion_matrix.png",
                    args.cross_pair_confusion_matrix_path,
                ),
            )
        else:
            if not args.eval_csv:
                raise ValueError(
                    "--eval_only requires --eval_csv, or both --eval_csv_hard and --eval_csv_cross_pair"
                )
            eval_from_csv_multiclass(
                eval_csv=args.eval_csv,
                model_dir=args.model_dir,
                mode=args.mode,
                save_predictions_csv=default_result_path(
                    eval_layout, "predictions.csv", args.save_predictions_csv
                ),
            )
            cm_path = default_plot_path(
                eval_layout, "confusion_matrix.png", args.confusion_matrix_path
            )
            if cm_path:
                bert = BertModel.from_pretrained(args.model_dir, add_pooling_layer=False)
                if bert.get_input_embeddings().weight.shape[0] != len(_bid.tokenizer):
                    bert.resize_token_embeddings(len(_bid.tokenizer))
                head = nn.Linear(bert.config.hidden_size, N_CLASSES)
                head.load_state_dict(
                    torch.load(Path(args.model_dir) / MC_CLS_HEAD_FILE, map_location="cpu")
                )
                eval_model = MulticlassConsistencyClassifier(bert, head).to(DEVICE)
                ds = CsvEvalDatasetMulticlass(args.eval_csv, mode=args.mode)
                plot_confusion_matrix_multiclass(eval_model, ds, cm_path)

    else:
        if args.balanced_error_weights and args.train_mode != "hard":
            print(
                "Warning: --balanced_error_weights only affects --train_mode hard; ignoring."
            )
        error_type_weights = resolve_error_type_weights(
            args.error_type_weights,
            use_balanced_preset=args.balanced_error_weights and args.train_mode == "hard",
        )
        if error_type_weights:
            print(f"Hard-negative generator sampling weights: {error_type_weights}")
        train_layout = setup_run(
            SCRIPT_ID,
            tags=_run_tags(),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            output_model_dir=args.output_model_dir,
            argv=current_argv(),
        )
        print(f"Run directory: {train_layout.root}")
        nf_swap_frac = (
            None
            if args.nf_swap_nf_to_disease_frac < 0
            else args.nf_swap_nf_to_disease_frac
        )
        saved_model_dir = train_multiclass(
            train_mode=args.train_mode,
            corruption_prob=args.corruption_prob,
            negatives_per_positive=args.negatives_per_positive,
            swap_instances=args.swap_instances,
            nf_labels_csv=args.nf_labels_csv,
            nf_swap_nf_to_disease_frac=nf_swap_frac,
            val_hard_eval_csv=args.val_hard_eval_csv,
            grad_clip=args.grad_clip,
            error_type_weights=error_type_weights,
            severity_jitter_prob=args.severity_jitter_prob,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name or train_layout.root.name,
            output_model_dir=str(train_layout.model),
        )
        # Optional post-training eval in the same run
        if args.eval_csv_hard and args.eval_csv_cross_pair:
            eval_benchmarks_multiclass(
                model_dir=saved_model_dir,
                eval_csv_hard=args.eval_csv_hard,
                eval_csv_cross_pair=args.eval_csv_cross_pair,
                hard_predictions_csv=default_result_path(
                    train_layout, "hard_predictions.csv", args.hard_predictions_csv
                ),
                cross_pair_predictions_csv=default_result_path(
                    train_layout, "cross_pair_predictions.csv", args.cross_pair_predictions_csv
                ),
                hard_confusion_matrix_path=default_plot_path(
                    train_layout, "hard_confusion_matrix.png", args.hard_confusion_matrix_path
                ),
                cross_pair_confusion_matrix_path=default_plot_path(
                    train_layout,
                    "cross_pair_confusion_matrix.png",
                    args.cross_pair_confusion_matrix_path,
                ),
            )
        elif args.eval_csv:
            eval_from_csv_multiclass(
                eval_csv=args.eval_csv,
                model_dir=saved_model_dir,
                mode=args.mode,
                save_predictions_csv=default_result_path(
                    train_layout, "predictions.csv", args.save_predictions_csv
                ),
            )
            cm_path = default_plot_path(
                train_layout, "confusion_matrix.png", args.confusion_matrix_path
            )
            if cm_path:
                bert = BertModel.from_pretrained(saved_model_dir, add_pooling_layer=False)
                if bert.get_input_embeddings().weight.shape[0] != len(_bid.tokenizer):
                    bert.resize_token_embeddings(len(_bid.tokenizer))
                head = nn.Linear(bert.config.hidden_size, N_CLASSES)
                head.load_state_dict(
                    torch.load(Path(saved_model_dir) / MC_CLS_HEAD_FILE, map_location="cpu")
                )
                eval_model = MulticlassConsistencyClassifier(bert, head).to(DEVICE)
                ds = CsvEvalDatasetMulticlass(args.eval_csv, mode=args.mode)
                plot_confusion_matrix_multiclass(eval_model, ds, cm_path)
