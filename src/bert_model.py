#bert from hugging face
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tokenizers.processors import TemplateProcessing
from torch.nn.functional import cosine_similarity
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)

try:
    from bert_architectures import (
        MODEL_SIZES,
        bert_mlm_config,
        default_mlm_batch_size,
        normalize_model_size,
        print_training_config,
    )
    from paths import DEFAULT_REPORTS_CSV, DEFAULT_TOKENIZER_PATH, init_pretrain_model_dir
    from report_split_io import (
        build_report_record,
        write_report_split_csv,
    )
except ImportError:
    _repo_root = Path(__file__).resolve().parents[1]
    import sys

    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from bert_architectures import (
        MODEL_SIZES,
        bert_mlm_config,
        default_mlm_batch_size,
        normalize_model_size,
        print_training_config,
    )
    from paths import DEFAULT_REPORTS_CSV, DEFAULT_TOKENIZER_PATH, init_pretrain_model_dir
    from report_split_io import (
        build_report_record,
        write_report_split_csv,
    )



def load_report_splits(reports_csv: str | Path | None = None):
    """Load findings/conclusion rows from a user CSV.

    Expected columns: Findings, Conclusion, and optionally split in {train,val,test}.
    Place the file at data/reports.csv, or set REPORTS_CSV / pass reports_csv.
    """
    data = pd.read_csv(reports_csv or DEFAULT_REPORTS_CSV)
    non_empty_findings = data["Findings"].fillna("").astype(str).str.strip().ne("")
    non_empty_conclusion = data["Conclusion"].fillna("").astype(str).str.strip().ne("")
    data = data[non_empty_findings | non_empty_conclusion]
    train_data = data[data["split"] == "train"] if "split" in data.columns else data
    test_data = data[data["split"] == "test"] if "split" in data.columns else data
    val_data = data[data["split"] == "val"] if "split" in data.columns else data
    print("total dataset size: ", len(data), "\ntrain dataset size: ", len(train_data))
    return train_data, test_data, val_data


def init_tokenizer(tokenizer_path: str):
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


class MlmTrainer(Trainer):
    """Trainer that can decouple LR schedule length from actual train length."""

    def __init__(self, scheduler_total_epochs: int | None = None, **kwargs):
        self.scheduler_total_epochs = scheduler_total_epochs
        super().__init__(**kwargs)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        scheduler_steps = num_training_steps
        if self.scheduler_total_epochs is not None:
            train_epochs = max(1, self.args.num_train_epochs)
            steps_per_epoch = max(1, num_training_steps // train_epochs)
            scheduler_steps = int(self.scheduler_total_epochs * steps_per_epoch)
            print(
                f"  LR schedule horizon:     {scheduler_steps} optimizer steps "
                f"({self.scheduler_total_epochs} epochs) while training "
                f"{num_training_steps} steps ({self.args.num_train_epochs} epochs)",
                flush=True,
            )
        super().create_optimizer_and_scheduler(num_training_steps=scheduler_steps)


def resolve_training_precision(precision: str | None) -> tuple[bool, bool]:
    """Return (fp16, bf16) for TrainingArguments."""
    if precision == "fp32":
        return False, False
    if precision == "bf16":
        use_cuda = torch.cuda.is_available()
        return False, use_cuda and torch.cuda.is_bf16_supported()
    if precision == "fp16":
        return torch.cuda.is_available(), False
    return torch.cuda.is_available(), False


def train_mlm(
    *,
    seq_length: int,
    model_size: str,
    output_model_dir: str,
    batch_size: int,
    epochs: int,
    tokenizer_path: str,
    learning_rate: float = 5e-4,
    warmup_steps: int = 0,
    precision: str | None = None,
    run_name: str | None = None,
    scheduler_total_epochs: int | None = None,
):
    train_data, test_data, val_data = load_report_splits()
    write_report_split_csv("train_text_path", train_data)
    write_report_split_csv("test_text_path", test_data)
    write_report_split_csv("val_text_path", val_data)

    dataset = load_dataset(
        "csv",
        data_files={"train": "train_text_path", "test": "test_text_path"},
    )
    tokenizer = init_tokenizer(tokenizer_path)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="longest",
            max_length=seq_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    norm_size = normalize_model_size(model_size)
    config = bert_mlm_config(len(tokenizer), seq_length, model_size)
    fp16, bf16 = resolve_training_precision(precision)
    resolved_run_name = run_name or f"bert_mlm_{norm_size}_mxlen_{seq_length}"
    print_training_config(
        stage="MLM",
        model_size=norm_size,
        seq_length=seq_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        output_dir=output_model_dir,
        config=config,
    )
    print(
        f"  warmup_steps:            {warmup_steps}",
        f"  precision:               {precision or 'fp16-if-cuda'}",
        f"  fp16/bf16:               {fp16}/{bf16}",
        f"  run_name:                {resolved_run_name}",
        f"  scheduler_total_epochs:  {scheduler_total_epochs or epochs}",
        flush=True,
    )
    model = BertForMaskedLM(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        run_name=resolved_run_name,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        evaluation_strategy="steps",
        eval_steps=500,
        fp16=fp16,
        bf16=bf16,
    )
    trainer = MlmTrainer(
        scheduler_total_epochs=scheduler_total_epochs,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_model_dir)
    print(f"Saved MLM model to: {output_model_dir}")
    return tokenizer, tokenized_dataset, train_data, test_data, training_args, data_collator


def run_post_training_eval(
    *,
    model_dir: str,
    tokenizer,
    tokenized_dataset,
    train_data,
    test_data,
    training_args,
    data_collator,
):
    texts = [build_report_record(row).text for _, row in train_data.iterrows()]
    test_texts = [build_report_record(row).text for _, row in test_data.iterrows()]

    eval_model = BertForMaskedLM.from_pretrained(model_dir)
    eval_model.eval()

    fill_mask = pipeline("fill-mask", model=eval_model, tokenizer=tokenizer)
    print(fill_mask("Extremely FDG-avid mid-oesophageal [MASK]."))
    print(fill_mask("Extremely FDG-avid mid-[MASK] tumour."))

    eval_trainer = Trainer(
        model=eval_model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    print(eval_trainer.evaluate())

    def normalize(x):
        return x / x.norm(dim=1, keepdim=True)

    def get_embeddings(texts_to_embed):
        if isinstance(texts_to_embed, str):
            texts_to_embed = [texts_to_embed]
        else:
            texts_to_embed = list(texts_to_embed)

        cleaned_texts = []
        for text in texts_to_embed:
            if pd.isna(text):
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(text).strip())

        inputs = tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        model_device = next(eval_model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = eval_model.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()

    def build_corpus_embeddings(texts_to_embed, batch_size=32):
        all_embeddings = []
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i : i + batch_size]
            all_embeddings.append(get_embeddings(batch))
        return torch.cat(all_embeddings, dim=0)

    corpus_embeddings = build_corpus_embeddings(test_texts)
    emb = corpus_embeddings.numpy()

    kmeans = KMeans(n_clusters=5).fit(emb)
    centers = kmeans.cluster_centers_
    for i, center in enumerate(centers):
        distances = np.linalg.norm(emb - center, axis=1)
        closest_idx = np.argsort(distances)[:5]
        print(f"\nCluster {i} representative texts:")
        for idx in closest_idx:
            print("-", texts[idx])
    print(np.bincount(kmeans.labels_))

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(emb)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=kmeans.labels_)
    plt.savefig("bert_plot_kmeans2.png")

    emb_2d = TSNE(n_components=2).fit_transform(emb)
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1])
    plt.savefig("bert_plot_tsne2.png")

    def retrieve(query, corpus_emb, top_k=5):
        q_emb = get_embeddings([query])
        corpus_emb = normalize(corpus_emb)
        q_emb = normalize(q_emb)
        sims = cosine_similarity(q_emb, corpus_emb, dim=1)
        return torch.topk(sims, k=top_k)

    query = test_texts[0]
    indices, scores = retrieve(query, corpus_embeddings)
    print("query:\n", query)
    for i, s in zip(indices, scores):
        print("Score:", s.item())
        print(texts[i])
        print("-" * 50)

    findings_texts = test_data["Findings"].fillna("").astype(str).str.strip().tolist()
    conclusion_texts = test_data["Conclusion"].fillna("").astype(str).str.strip().tolist()
    findings_emb = normalize(get_embeddings(findings_texts))
    conclusion_emb = normalize(get_embeddings(conclusion_texts))
    sims = cosine_similarity(findings_emb, conclusion_emb, dim=1)
    print("Mean similarity:", sims.mean().item())

    def alignment_accuracy(findings_emb_local, conclusion_emb_local, top_k=1):
        correct = 0
        for i in range(len(findings_emb_local)):
            pair_sims = cosine_similarity(
                findings_emb_local[i].unsqueeze(0),
                conclusion_emb_local,
                dim=1,
            )
            topk = torch.topk(pair_sims, k=top_k).indices
            if i in topk:
                correct += 1
        return correct / len(findings_emb_local)

    for top_k in (1, 5, 10, 100):
        acc = alignment_accuracy(findings_emb, conclusion_emb, top_k=top_k)
        print(f"Top-{top_k} accuracy:", acc)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT MLM pretraining on radiology reports.")
    parser.add_argument("--seq_length", type=int, default=600, choices=(512, 600, 768))
    parser.add_argument(
        "--model_size",
        type=str,
        default="big",
        help=f"One of {MODEL_SIZES} or layer count 4/6/8/12.",
    )
    parser.add_argument("--output_model_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=DEFAULT_TOKENIZER_PATH,
    )
    parser.add_argument(
        "--skip_post_eval",
        action="store_true",
        help="Skip fill-mask, clustering, and embedding probes after training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Peak AdamW learning rate.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Linear LR warmup steps.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp16", "bf16", "fp32"),
        default=None,
        help="Mixed/full precision. Default: fp16 when CUDA is available.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional wandb / Trainer run name.",
    )
    parser.add_argument(
        "--scheduler_total_epochs",
        type=int,
        default=None,
        help=(
            "Total epochs assumed by the linear LR scheduler. "
            "Defaults to --epochs. Use 50 for short probes so LR decay matches full pretraining."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_model_dir = args.output_model_dir or str(
        init_pretrain_model_dir(args.seq_length, model_size=args.model_size)
    )
    batch_size = args.batch_size or default_mlm_batch_size(args.model_size)
    Path(output_model_dir).mkdir(parents=True, exist_ok=True)

    tokenizer, tokenized_dataset, train_data, test_data, training_args, data_collator = train_mlm(
        seq_length=args.seq_length,
        model_size=args.model_size,
        output_model_dir=output_model_dir,
        batch_size=batch_size,
        epochs=args.epochs,
        tokenizer_path=args.tokenizer_path,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        precision=args.precision,
        run_name=args.run_name,
        scheduler_total_epochs=args.scheduler_total_epochs,
    )

    if not args.skip_post_eval:
        run_post_training_eval(
            model_dir=output_model_dir,
            tokenizer=tokenizer,
            tokenized_dataset=tokenized_dataset,
            train_data=train_data,
            test_data=test_data,
            training_args=training_args,
            data_collator=data_collator,
        )


if __name__ == "__main__":
    main()
