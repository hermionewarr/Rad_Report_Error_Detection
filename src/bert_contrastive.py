
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel

try:
    from paths import (
        DEFAULT_TOKENIZER_PATH,
        checkpoint_path,
        current_argv,
        data_path,
        setup_run,
    )
    from report_split_io import load_findings_conclusion_pairs
except ImportError:
    _repo_root = Path(__file__).resolve().parents[1]
    import sys

    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from paths import (
        DEFAULT_TOKENIZER_PATH,
        checkpoint_path,
        current_argv,
        data_path,
        setup_run,
    )
    from report_split_io import load_findings_conclusion_pairs

#MODEL_PATH = checkpoint_path("bert-pretrained2")
MODEL_PATH = checkpoint_path("bert-pretrained_mxlen_512", require_exists=False)
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH
TRAIN_FILE = data_path("train_text_path")
TEST_FILE = data_path("test_text_path")
VAL_FILE = data_path("val_text_path")
MAX_LENGTH = 512 #128
BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-5
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_ID = "bert-contrastive"
RUN_CATEGORY = "pretraining"
DEFAULT_CHECKPOINT = "bert-contrastive_bs128_090526"
tokenizer = None


class FindingsConclusionDataset(Dataset):
    def __init__(self, file_path):
        self.pairs = load_findings_conclusion_pairs(file_path)
        print(f"Loaded {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch):
    findings, conclusions = zip(*batch)
    f_inputs = tokenizer(
        list(findings),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    c_inputs = tokenizer(
        list(conclusions),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return f_inputs, c_inputs


def contrastive_loss(f_emb, c_emb, temperature=0.05):
    f_emb = F.normalize(f_emb, dim=1)
    c_emb = F.normalize(c_emb, dim=1)
    logits = torch.matmul(f_emb, c_emb.T) / temperature
    labels = torch.arange(len(f_emb), device=f_emb.device)
    loss_f = F.cross_entropy(logits, labels)
    loss_c = F.cross_entropy(logits.T, labels)
    return (loss_f + loss_c) / 2


def compute_retrieval_metrics(findings_emb, conclusions_emb, top_k=(1, 5, 10)):
    findings_emb = F.normalize(findings_emb, dim=1)
    conclusions_emb = F.normalize(conclusions_emb, dim=1)
    sims = torch.matmul(findings_emb, conclusions_emb.T)
    labels = torch.arange(len(findings_emb), device=findings_emb.device)
    out = {}
    for k in top_k:
        topk = torch.topk(sims, k=k, dim=1).indices
        out[f"top_{k}"] = (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return out


def full_dataset_eval(model, eval_dataloader, device, temperature=0.05):
    model.eval()
    all_f_emb, all_c_emb, losses = [], [], []
    with torch.no_grad():
        for f_inputs, c_inputs in eval_dataloader:
            f_inputs = {k: v.to(device) for k, v in f_inputs.items()}
            c_inputs = {k: v.to(device) for k, v in c_inputs.items()}
            f_emb = model(**f_inputs).last_hidden_state[:, 0, :]
            c_emb = model(**c_inputs).last_hidden_state[:, 0, :]
            losses.append(contrastive_loss(f_emb, c_emb, temperature=temperature).item())
            all_f_emb.append(f_emb)
            all_c_emb.append(c_emb)

    all_f_emb = torch.cat(all_f_emb, dim=0)
    all_c_emb = torch.cat(all_c_emb, dim=0)
    return {
        "loss": float(np.mean(losses)),
        **compute_retrieval_metrics(all_f_emb, all_c_emb),
    }


def embed_texts(model, texts, batch_size=BATCH_SIZE):
    all_emb = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            emb = model(**inputs).last_hidden_state[:, 0, :]
            all_emb.append(F.normalize(emb, dim=1).cpu())
    return torch.cat(all_emb, dim=0)


def build_binary_eval_pairs(pairs, seed=42):
    rng = random.Random(seed)
    findings, conclusions, labels = [], [], []
    n = len(pairs)
    for f_text, c_text in pairs:
        findings.append(f_text)
        conclusions.append(c_text)
        labels.append(1)
    for i, (f_text, _) in enumerate(pairs):
        j = rng.randrange(n)
        while j == i:
            j = rng.randrange(n)
        findings.append(f_text)
        conclusions.append(pairs[j][1])
        labels.append(0)

    # Shuffle positives/negatives while preserving alignment, deterministic by seed.
    rows = list(zip(findings, conclusions, labels))
    rng.shuffle(rows)
    findings, conclusions, labels = zip(*rows)
    return list(findings), list(conclusions), torch.tensor(labels, dtype=torch.long)


def binary_metrics_from_scores(scores, labels, threshold):
    preds = (scores >= threshold).long()
    acc = (preds == labels).float().mean().item()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    tp0 = ((preds == 0) & (labels == 0)).sum().item()
    fp0 = ((preds == 0) & (labels == 1)).sum().item()
    fn0 = ((preds == 1) & (labels == 0)).sum().item()
    tn0 = ((preds == 1) & (labels == 1)).sum().item()
    p0 = tp0 / (tp0 + fp0 + 1e-8)
    r0 = tp0 / (tp0 + fn0 + 1e-8)
    f10 = 2 * p0 * r0 / (p0 + r0 + 1e-8)
    return {
        "threshold": float(threshold),
        "acc": acc,
        "precision_consistent": precision,
        "recall_consistent": recall,
        "f1_consistent": f1,
        "precision_inconsistent": p0,
        "recall_inconsistent": r0,
        "f1_inconsistent": f10,
        "tp0": int(tp0),
        "fp0": int(fp0),
        "fn0": int(fn0),
        "tn0": int(tn0),
    }


def evaluate_binary_consistency(model, pairs, threshold, seed=42):
    findings, conclusions, labels = build_binary_eval_pairs(pairs, seed=seed)
    f_emb = embed_texts(model, findings)
    c_emb = embed_texts(model, conclusions)
    scores = torch.sum(f_emb * c_emb, dim=1)
    return binary_metrics_from_scores(scores, labels, threshold)


def tune_threshold_on_pairs(model, pairs, seed=42, steps=400):
    findings, conclusions, labels = build_binary_eval_pairs(pairs, seed=seed)
    f_emb = embed_texts(model, findings)
    c_emb = embed_texts(model, conclusions)
    scores = torch.sum(f_emb * c_emb, dim=1)
    thresholds = torch.linspace(float(scores.min()), float(scores.max()), steps=steps)
    best_metrics, best_score = None, -1.0
    for threshold in thresholds:
        metrics = binary_metrics_from_scores(scores, labels, float(threshold.item()))
        if metrics["acc"] > best_score:
            best_score = metrics["acc"]
            best_metrics = metrics
    return best_metrics


def plot_confusion_matrix_inconsistent_positive(metrics, output_path, title):
    matrix = [
        [int(metrics["tp0"]), int(metrics["fn0"])],
        [int(metrics["fp0"]), int(metrics["tn0"])],
    ]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], labels=["Pred: Inconsistent", "Pred: Consistent"])
    ax.set_yticks([0, 1], labels=["True: Inconsistent", "True: Consistent"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def run_binary_evaluations(
    model,
    test_pairs,
    val_pairs,
    seed=42,
    fixed_cm_path="contrastive_binary_confusion_matrix.png",
    tuned_cm_path="contrastive_val_tuned_confusion_matrix.png",
):
    fixed_zero_metrics = evaluate_binary_consistency(model, test_pairs, 0.0, seed=seed)
    plot_confusion_matrix_inconsistent_positive(
        fixed_zero_metrics, fixed_cm_path, "Contrastive Confusion Matrix (Threshold = 0.0)"
    )
    tuned_val_metrics = tune_threshold_on_pairs(model, val_pairs, seed=seed)
    tuned_test_metrics = evaluate_binary_consistency(
        model, test_pairs, tuned_val_metrics["threshold"], seed=seed
    )
    plot_confusion_matrix_inconsistent_positive(
        tuned_test_metrics, tuned_cm_path, "Contrastive Confusion Matrix (VAL-Tuned Threshold)"
    )
    print("Binary eval (test, threshold=0.0):", fixed_zero_metrics)
    print("Binary eval (val tuned threshold):", tuned_val_metrics)
    print("Binary eval (test, val-tuned threshold):", tuned_test_metrics)
    return fixed_zero_metrics, tuned_val_metrics, tuned_test_metrics


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.cls_token_id),
            ("[SEP]", tokenizer.sep_token_id),
        ],
    )
    return tokenizer


def init_wandb(run_name, contrastive_temperature=0.05):
    wandb.init(
        project="bert-contrastive",
        name=run_name,
        config={
            "model": MODEL_PATH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "max_length": MAX_LENGTH,
            "contrastive_temperature": contrastive_temperature,
        },
    )


def train_and_eval(output_model_dir=None, contrastive_temperature=0.05, run_layout=None):
    if not output_model_dir:
        raise ValueError("output_model_dir is required for training")
    init_wandb(Path(output_model_dir).parent.name, contrastive_temperature=contrastive_temperature)
    init_tokenizer(TOKENIZER_PATH)
    model = BertModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))

    dataset = FindingsConclusionDataset(TRAIN_FILE)
    test_dataset = FindingsConclusionDataset(TEST_FILE)
    val_dataset = FindingsConclusionDataset(VAL_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    global_step = 0
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for f_inputs, c_inputs in dataloader:
            f_inputs = {k: v.to(DEVICE) for k, v in f_inputs.items()}
            c_inputs = {k: v.to(DEVICE) for k, v in c_inputs.items()}
            f_emb = model(**f_inputs).last_hidden_state[:, 0, :]
            c_emb = model(**c_inputs).last_hidden_state[:, 0, :]
            loss = contrastive_loss(f_emb, c_emb, temperature=contrastive_temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if global_step % 10 == 0:
                wandb.log({"train/loss": loss.item()}, step=global_step)
                with torch.no_grad():
                    sims = torch.sum(
                        F.normalize(f_emb, dim=1) * F.normalize(c_emb, dim=1), dim=1
                    ).mean()
                wandb.log({"train/positive_similarity": sims.item()}, step=global_step)
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        wandb.log({"train/epoch_loss": avg_loss}, step=global_step)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        model.eval()
        eval_batch = next(iter(eval_dataloader))
        f_inputs, c_inputs = eval_batch
        f_inputs = {k: v.to(DEVICE) for k, v in f_inputs.items()}
        c_inputs = {k: v.to(DEVICE) for k, v in c_inputs.items()}
        with torch.no_grad():
            f_emb = model(**f_inputs).last_hidden_state[:, 0, :]
            c_emb = model(**c_inputs).last_hidden_state[:, 0, :]
            eval_loss = contrastive_loss(f_emb, c_emb, temperature=contrastive_temperature)
        metrics = compute_retrieval_metrics(f_emb, c_emb)
        wandb.log(
            {
                "eval/loss": eval_loss.item(),
                "eval/top_1": metrics["top_1"],
                "eval/top_5": metrics["top_5"],
                "eval/top_10": metrics["top_10"],
                "epoch": epoch,
            },
            step=global_step,
        )
        model.train()

        if epoch == EPOCHS - 1:
            final_eval = full_dataset_eval(
                model=model,
                eval_dataloader=eval_dataloader,
                device=DEVICE,
                temperature=contrastive_temperature,
            )
            print("Final full-dataset retrieval:", final_eval)
            plots = run_layout.plots if run_layout is not None else Path(".")
            fixed_zero_metrics, tuned_val_metrics, tuned_test_metrics = run_binary_evaluations(
                model=model,
                test_pairs=test_dataset.pairs,
                val_pairs=val_dataset.pairs,
                seed=SEED,
                fixed_cm_path=str(plots / "contrastive_binary_confusion_matrix.png"),
                tuned_cm_path=str(plots / "contrastive_val_tuned_confusion_matrix.png"),
            )
            wandb.log(
                {
                    "binary_eval/fixed_zero_acc": fixed_zero_metrics["acc"],
                    "binary_eval/fixed_zero_f1_inconsistent": fixed_zero_metrics["f1_inconsistent"],
                    "binary_eval/val_tuned_threshold": tuned_val_metrics["threshold"],
                    "binary_eval/val_best_acc": tuned_val_metrics["acc"],
                    "binary_eval/tuned_test_acc": tuned_test_metrics["acc"],
                    "binary_eval/tuned_test_f1_inconsistent": tuned_test_metrics["f1_inconsistent"],
                },
                step=global_step,
            )

    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)


def eval_only(model_path, tokenizer_path):
    init_tokenizer(tokenizer_path)
    model = BertModel.from_pretrained(model_path).to(DEVICE)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    test_dataset = FindingsConclusionDataset(TEST_FILE)
    val_dataset = FindingsConclusionDataset(VAL_FILE)
    run_binary_evaluations(
        model=model,
        test_pairs=test_dataset.pairs,
        val_pairs=val_dataset.pairs,
        seed=SEED,
        fixed_cm_path="contrastive_binary_confusion_matrix_eval_only.png",
        tuned_cm_path="contrastive_val_tuned_confusion_matrix_eval_only.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--contrastive_temperature", type=float, default=0.05)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_tokenizer_path", type=str, default=TOKENIZER_PATH)
    args = parser.parse_args()
    args.eval_model_path = args.eval_model_path or checkpoint_path(DEFAULT_CHECKPOINT)

    if args.eval_only:
        eval_layout = setup_run(
            SCRIPT_ID,
            tags=("eval",),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            argv=current_argv(),
        )
        print(f"Run outputs: {eval_layout.root}")
        eval_only(args.eval_model_path, args.eval_tokenizer_path)
    else:
        train_layout = setup_run(
            SCRIPT_ID,
            tags=(f"t{args.contrastive_temperature:g}",),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            output_model_dir=args.output_model_dir,
            argv=current_argv(),
        )
        print(f"Run directory: {train_layout.root}")
        train_and_eval(
            output_model_dir=str(train_layout.model),
            contrastive_temperature=args.contrastive_temperature,
            run_layout=train_layout,
        )