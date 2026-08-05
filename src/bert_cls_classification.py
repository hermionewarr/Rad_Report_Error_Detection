import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import wandb
from tokenizers.processors import TemplateProcessing
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel

try:
    from bert_architectures import default_cls_batch_size, print_training_config
    from paths import (
        DEFAULT_TOKENIZER_PATH,
        checkpoint_path,
        current_argv,
        data_path,
        default_plot_path,
        init_cls_model_dir,
        init_pretrain_model_dir,
        setup_run,
    )
    from report_split_io import load_findings_conclusion_pairs
except ImportError:
    _repo_root = Path(__file__).resolve().parents[1]
    import sys

    if str(_repo_root) not in sys.path:
        sys.path.append(str(_repo_root))
    from bert_architectures import default_cls_batch_size, print_training_config
    from paths import (
        DEFAULT_TOKENIZER_PATH,
        checkpoint_path,
        current_argv,
        data_path,
        default_plot_path,
        init_cls_model_dir,
        init_pretrain_model_dir,
        setup_run,
    )
    from report_split_io import load_findings_conclusion_pairs

MAX_LENGTH = 600 #768 #512 #256 #128
MODEL_PATH = str(init_pretrain_model_dir(MAX_LENGTH))
TOKENIZER_PATH = DEFAULT_TOKENIZER_PATH
TRAIN_FILE = data_path("train_text_path")
TEST_FILE = data_path("test_text_path")
VAL_FILE = data_path("val_text_path")

BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-5
SEED = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_ID = "bert-cls"
RUN_CATEGORY = "pretraining"
DEFAULT_CHECKPOINT = "bert-cls-binary"
CLS_HEAD_FILE = "cls_classifier.pt"
tokenizer = None
SCRATCH_INIT = False


def configure_training(
    *,
    seq_length: int,
    model_size: str = "small",
    init_checkpoint: str | None = None,
    scratch_init: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    global MAX_LENGTH, MODEL_PATH, EPOCHS, BATCH_SIZE, SCRATCH_INIT
    MAX_LENGTH = seq_length
    SCRATCH_INIT = scratch_init
    if scratch_init:
        MODEL_PATH = f"<scratch:{model_size}>"
    elif init_checkpoint:
        init_path = Path(init_checkpoint)
        MODEL_PATH = (
            str(init_path.resolve())
            if init_path.is_dir()
            else checkpoint_path(init_checkpoint)
        )
    else:
        MODEL_PATH = str(init_pretrain_model_dir(seq_length, model_size=model_size))
    if epochs is not None:
        EPOCHS = epochs
    if batch_size is not None:
        BATCH_SIZE = batch_size


FINDINGS_SPECIAL = "[Findings]"
CONCLUSION_SPECIAL = "[Conclusion]"


def format_joint_report(findings: str, conclusion: str) -> str:
    return (
        f"{tokenizer.cls_token} {FINDINGS_SPECIAL} {findings} "
        f"{CONCLUSION_SPECIAL} {conclusion} {tokenizer.sep_token}"
    )


class FindingsConclusionDataset(Dataset):
    def __init__(self, file_path):
        self.pairs = load_findings_conclusion_pairs(file_path)
        print(f"Loaded {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn_cls_classifier(batch, negative_prob=0.5):
    findings, conclusions = zip(*batch)
    findings = list(findings)
    conclusions = list(conclusions)
    labels = []
    texts = []
    n = len(batch)
    for i, findings_text in enumerate(findings):
        do_shuffle = random.random() < negative_prob and n > 1
        chosen_conclusion = conclusions[i]
        if do_shuffle:
            j = random.randrange(n)
            while j == i:
                j = random.randrange(n)
            chosen_conclusion = conclusions[j]
            labels.append(0.0)
        else:
            labels.append(1.0)
        texts.append(format_joint_report(findings_text, chosen_conclusion))

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded["labels"] = torch.tensor(labels, dtype=torch.float32)
    return encoded


def single_cls_logits(model, classifier_head, input_batch):
    outputs = model(**input_batch)
    cls_repr = outputs.last_hidden_state[:, 0, :]
    return classifier_head(cls_repr).squeeze(-1)


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

    rows = list(zip(findings, conclusions, labels))
    rng.shuffle(rows)
    findings, conclusions, labels = zip(*rows)
    return list(findings), list(conclusions), torch.tensor(labels, dtype=torch.long)


def predict_scores(model, classifier_head, findings, conclusions, batch_size=BATCH_SIZE):
    scores = []
    model.eval()
    classifier_head.eval()
    with torch.no_grad():
        for i in range(0, len(findings), batch_size):
            f_batch = findings[i : i + batch_size]
            c_batch = conclusions[i : i + batch_size]
            texts = [format_joint_report(f, c) for f, c in zip(f_batch, c_batch)]
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            logits = single_cls_logits(model, classifier_head, inputs)
            scores.append((2.0 * torch.sigmoid(logits) - 1.0).cpu())
    return torch.cat(scores, dim=0)


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


def evaluate_binary_consistency(model, classifier_head, pairs, threshold, seed=42):
    findings, conclusions, labels = build_binary_eval_pairs(pairs, seed=seed)
    scores = predict_scores(model, classifier_head, findings, conclusions)
    return binary_metrics_from_scores(scores, labels, threshold)


def tune_threshold_on_pairs(model, classifier_head, pairs, seed=42, steps=400):
    findings, conclusions, labels = build_binary_eval_pairs(pairs, seed=seed)
    scores = predict_scores(model, classifier_head, findings, conclusions)
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
    classifier_head,
    test_pairs,
    val_pairs,
    seed=42,
    fixed_cm_path="cls_binary_confusion_matrix.png",
    tuned_cm_path="cls_val_tuned_confusion_matrix.png",
):
    fixed_zero_metrics = evaluate_binary_consistency(
        model=model,
        classifier_head=classifier_head,
        pairs=test_pairs,
        threshold=0.0,
        seed=seed,
    )
    plot_confusion_matrix_inconsistent_positive(
        fixed_zero_metrics, fixed_cm_path, "CLS Confusion Matrix (Threshold = 0.0)"
    )
    tuned_val_metrics = tune_threshold_on_pairs(
        model=model, classifier_head=classifier_head, pairs=val_pairs, seed=seed
    )
    tuned_test_metrics = evaluate_binary_consistency(
        model=model,
        classifier_head=classifier_head,
        pairs=test_pairs,
        threshold=tuned_val_metrics["threshold"],
        seed=seed,
    )
    plot_confusion_matrix_inconsistent_positive(
        tuned_test_metrics, tuned_cm_path, "CLS Confusion Matrix (VAL-Tuned Threshold)"
    )
    print("Binary eval (test, threshold=0.0):", fixed_zero_metrics)
    print("Binary eval (val tuned threshold):", tuned_val_metrics)
    print("Binary eval (test, val-tuned threshold):", tuned_test_metrics)
    return fixed_zero_metrics, tuned_val_metrics, tuned_test_metrics


def init_tokenizer(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    extra = []
    vocab = tokenizer.get_vocab()
    for tok in (FINDINGS_SPECIAL, CONCLUSION_SPECIAL):
        if tok not in vocab:
            extra.append(tok)
    if extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[],
    )
    return tokenizer


def init_wandb(run_name, negative_prob=0.5, model_config=None):
    cfg = {
        "model": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "max_length": MAX_LENGTH,
        "single_cls_negative_prob": negative_prob,
    }
    if model_config is not None:
        cfg.update(
            {
                "hidden_size": model_config.hidden_size,
                "num_hidden_layers": model_config.num_hidden_layers,
                "num_attention_heads": model_config.num_attention_heads,
                "intermediate_size": model_config.intermediate_size,
            }
        )
    wandb.init(
        project="bert-cls-binary",
        name=run_name,
        config=cfg,
    )


def train_and_eval(
    output_model_dir=None,
    single_cls_negative_prob=0.5,
    negative_sampler="batch",
    run_layout=None,
    model_size: str = "small",
):
    if not output_model_dir:
        raise ValueError("output_model_dir is required for training")
    init_tokenizer(TOKENIZER_PATH)

    if SCRATCH_INIT:
        try:
            from bert_architectures import bert_mlm_config
        except ImportError:
            from bert_architectures import bert_mlm_config  # type: ignore[no-redef]
        model = BertModel(bert_mlm_config(len(tokenizer), MAX_LENGTH, model_size)).to(DEVICE)
    else:
        model = BertModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    classifier_head = torch.nn.Linear(model.config.hidden_size, 1).to(DEVICE)

    print_training_config(
        stage="CLS",
        model_size=model_size,
        seq_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LR,
        init_path=MODEL_PATH,
        output_dir=output_model_dir,
        config=model.config,
    )
    init_wandb(
        Path(output_model_dir).parent.name,
        negative_prob=single_cls_negative_prob,
        model_config=model.config,
    )

    dataset = FindingsConclusionDataset(TRAIN_FILE)
    test_dataset = FindingsConclusionDataset(TEST_FILE)
    val_dataset = FindingsConclusionDataset(VAL_FILE)

    train_collate = lambda batch: collate_fn_cls_classifier(
        batch, negative_prob=single_cls_negative_prob
    )
    eval_collate = lambda batch: collate_fn_cls_classifier(
        batch, negative_prob=single_cls_negative_prob
    )

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_collate
    )
    eval_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=eval_collate
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier_head.parameters()),
        lr=LR,
    )
    global_step = 0
    model.train()
    classifier_head.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = single_cls_logits(model, classifier_head, inputs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if global_step % 10 == 0:
                preds = (torch.sigmoid(logits) >= 0.5).float()
                batch_acc = (preds == labels).float().mean().item()
                wandb.log(
                    {"train/loss": loss.item(), "train/batch_acc": batch_acc},
                    step=global_step,
                )
            global_step += 1

        avg_loss = total_loss / len(dataloader)
        wandb.log({"train/epoch_loss": avg_loss}, step=global_step)
        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

        model.eval()
        classifier_head.eval()
        eval_batch = next(iter(eval_dataloader))
        labels = eval_batch.pop("labels").to(DEVICE)
        inputs = {k: v.to(DEVICE) for k, v in eval_batch.items()}
        with torch.no_grad():
            logits = single_cls_logits(model, classifier_head, inputs)
            eval_loss = F.binary_cross_entropy_with_logits(logits, labels)
            eval_acc = ((torch.sigmoid(logits) >= 0.5).float() == labels).float().mean()
        wandb.log(
            {"eval/loss": eval_loss.item(), "eval/acc": eval_acc.item(), "epoch": epoch},
            step=global_step,
        )
        model.train()
        classifier_head.train()

        if epoch == EPOCHS - 1:
            plots = run_layout.plots if run_layout is not None else Path(".")
            fixed_zero_metrics, tuned_val_metrics, tuned_test_metrics = run_binary_evaluations(
                model=model,
                classifier_head=classifier_head,
                test_pairs=test_dataset.pairs,
                val_pairs=val_dataset.pairs,
                seed=SEED,
                fixed_cm_path=str(plots / "cls_binary_confusion_matrix.png"),
                tuned_cm_path=str(plots / "cls_val_tuned_confusion_matrix.png"),
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
    torch.save(classifier_head.state_dict(), Path(output_model_dir) / CLS_HEAD_FILE)
    print(f"Saved model to: {output_model_dir}")


def eval_only(model_path, tokenizer_path):
    init_tokenizer(tokenizer_path)
    model = BertModel.from_pretrained(model_path).to(DEVICE)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    classifier_head = torch.nn.Linear(model.config.hidden_size, 1).to(DEVICE)
    classifier_path = Path(model_path) / CLS_HEAD_FILE
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier head: {classifier_path}")
    classifier_head.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    classifier_head.eval()

    test_dataset = FindingsConclusionDataset(TEST_FILE)
    val_dataset = FindingsConclusionDataset(VAL_FILE)
    run_binary_evaluations(
        model=model,
        classifier_head=classifier_head,
        test_pairs=test_dataset.pairs,
        val_pairs=val_dataset.pairs,
        seed=SEED,
        fixed_cm_path="cls_binary_confusion_matrix_eval_only.png",
        tuned_cm_path="cls_val_tuned_confusion_matrix_eval_only.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--single_cls_negative_prob", type=float, default=0.5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--eval_tokenizer_path", type=str, default=TOKENIZER_PATH)
    parser.add_argument("--seq_length", type=int, default=MAX_LENGTH, choices=(512, 600, 768))
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        help="One of small/big/8L/12L (or 4/6/8/12). Used to resolve MLM init checkpoint.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional MLM checkpoint name/path. Overrides --seq_length/--model_size.",
    )
    parser.add_argument(
        "--scratch_init",
        action="store_true",
        help="Random-init BERT backbone (CLS pretrain without MLM). Ignored with --init_checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    cls_batch_size = args.batch_size or default_cls_batch_size(args.model_size)
    if args.scratch_init and args.init_checkpoint:
        parser.error("--scratch_init cannot be combined with --init_checkpoint")
    configure_training(
        seq_length=args.seq_length,
        model_size=args.model_size,
        init_checkpoint=args.init_checkpoint,
        scratch_init=args.scratch_init,
        epochs=args.epochs,
        batch_size=cls_batch_size,
    )
    args.eval_model_path = args.eval_model_path or checkpoint_path(DEFAULT_CHECKPOINT)

    default_cls_output = str(init_cls_model_dir(args.seq_length, model_size=args.model_size))

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
            tags=(f"neg{args.single_cls_negative_prob:g}",),
            run_name=args.run_name,
            run_category=RUN_CATEGORY,
            output_model_dir=args.output_model_dir or default_cls_output,
            argv=current_argv(),
        )
        print(f"Run directory: {train_layout.root}")
        train_and_eval(
            output_model_dir=str(train_layout.model),
            single_cls_negative_prob=args.single_cls_negative_prob,
            run_layout=train_layout,
            model_size=args.model_size,
        )
