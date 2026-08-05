"""BERT size presets for MLM pretraining and downstream CLS init."""
from __future__ import annotations

from transformers import BertConfig

MODEL_SIZES = ("small", "big", "8L", "12L")

_SIZE_ALIASES = {
    "4": "small",
    "small": "small",
    "6": "big",
    "big": "big",
    "8": "8L",
    "8l": "8L",
    "8L": "8L",
    "medium": "8L",
    "12": "12L",
    "12l": "12L",
    "12L": "12L",
    "xl": "12L",
    "base": "12L",
}


def normalize_model_size(model_size: str) -> str:
    key = str(model_size).strip()
    norm = _SIZE_ALIASES.get(key.lower())
    if norm is None:
        raise ValueError(
            f"model_size must be one of {MODEL_SIZES} or layer count 4/6/8/12, got {model_size!r}"
        )
    return norm


def model_size_suffix(model_size: str) -> str:
    norm = normalize_model_size(model_size)
    if norm == "small":
        return ""
    if norm == "big":
        return "_big"
    if norm == "8L":
        return "_8L"
    return "_12L"


def bert_mlm_config(
    vocab_size: int,
    max_length: int,
    model_size: str = "small",
) -> BertConfig:
    norm = normalize_model_size(model_size)
    presets = {
        "small": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
        },
        "big": {
            "hidden_size": 384,
            "num_hidden_layers": 6,
            "num_attention_heads": 6,
            "intermediate_size": 1536,
        },
        "8L": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
        },
        "12L": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
    }
    return BertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_length,
        **presets[norm],
    )


def default_mlm_batch_size(model_size: str) -> int:
    norm = normalize_model_size(model_size)
    if norm == "12L":
        return 16
    return 32


def default_cls_batch_size(model_size: str) -> int:
    norm = normalize_model_size(model_size)
    if norm == "12L":
        return 64
    return 128


def format_bert_config_lines(config: BertConfig) -> list[str]:
    head_dim = config.hidden_size // config.num_attention_heads
    return [
        f"  hidden_size:             {config.hidden_size}",
        f"  num_hidden_layers:       {config.num_hidden_layers}",
        f"  num_attention_heads:     {config.num_attention_heads}",
        f"  head_dim:                {head_dim}",
        f"  intermediate_size:       {config.intermediate_size}",
        f"  max_position_embeddings: {config.max_position_embeddings}",
        f"  vocab_size:              {config.vocab_size}",
    ]


def print_training_config(
    *,
    stage: str,
    config: BertConfig,
    batch_size: int,
    epochs: int,
    model_size: str | None = None,
    seq_length: int | None = None,
    init_path: str | None = None,
    output_dir: str | None = None,
    learning_rate: float | None = None,
) -> None:
    print("=" * 72)
    print(f"{stage} training config")
    if model_size is not None:
        print(f"  model_size:              {normalize_model_size(model_size)}")
    if seq_length is not None:
        print(f"  seq_length:              {seq_length}")
    print(f"  batch_size:              {batch_size}")
    print(f"  epochs:                  {epochs}")
    if learning_rate is not None:
        print(f"  learning_rate:           {learning_rate}")
    if init_path is not None:
        print(f"  init_checkpoint:         {init_path}")
    if output_dir is not None:
        print(f"  output_dir:              {output_dir}")
    for line in format_bert_config_lines(config):
        print(line)
    print("=" * 72, flush=True)
