# Local LLM baseline setup (privacy-safe)

Evaluates open-weight instruct models **on your machine only**. Report text is sent to `127.0.0.1` (vLLM/Ollama), not to cloud APIs.

## 1. Create conda environment

From the project root:

```bash
conda env create -f environment-llm-eval.yml
conda activate Rad_Report_Error_Detection-LLM
```

### What gets installed

| Package | Purpose |
|---------|---------|
| **Python 3.11** | Runtime |
| **vllm** | Local OpenAI-compatible server + GPU inference |
| **huggingface_hub** | Download open-weight models to `~/.cache/huggingface` |
| **hf_transfer** | Optional faster downloads (`export HF_HUB_ENABLE_HF_TRANSFER=1`) |

vLLM installs **PyTorch + CUDA** wheels automatically via pip.

### Host requirements

- NVIDIA driver + GPU (authors used RTX 6000 / A6000 — suitable for 70B with AWQ)
- Enough disk for model weights (~40–80 GB per 70B quantized checkpoint)
- For **gated** models (Llama): `huggingface-cli login` once

### Verify install

```bash
conda activate Rad_Report_Error_Detection-LLM
python -c "import vllm; print('vllm', vllm.__version__)"
python -m vllm.entrypoints.openai.api_server --help
```

## 2. Edit model list (optional)

Default three models (different families, pre-registered before runs):

`config/llm_models_default.json`

Change `model` Hugging Face IDs or `quantization` (`awq`, `gptq`, or `null`) per model.

**Important:** `--quantization awq` requires an **AWQ checkpoint** on Hugging Face (model name usually contains `AWQ`), not the base Instruct repo. Example:

| Wrong | Right |
|-------|-------|
| `Qwen/Qwen2.5-72B-Instruct` + `awq` | `Qwen/Qwen2.5-72B-Instruct-AWQ` + `awq` |

Per-model optional fields in JSON: `dtype` (`float16` for AWQ), `trust_remote_code`, `enforce_eager` (helps if OOM on 48GB), `tensor_parallel_size`.

Per-model GPU override (optional): `"cuda_visible_devices": "1"` on one entry.

Suite-wide GPU: `--gpu 1` on the runner command line.

## 3. Paper eval (single-swap, all error types)

Recommended eval CSV for the paper:

`data/eval_datasets/hard_negative_eval_v6_single_swap.csv`

Use **`--mode hard`** for the full v6 hard-negative suite (laterality, site, severity, NF vs disease, **cross_pair_mismatch**, plus consistent). LLM reports use **`eval_summary.txt`**.

```bash
cd /path/to/Rad_Report_Error_Detection
conda activate Rad_Report_Error_Detection-LLM

python scripts/run_llm_baselines.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v6_single_swap.csv \
  --mode hard \
  --suite_name single-swap-errors \
  --manage_vllm \
  --gpu 1
```

Suite output layout (`trained_models/runs/llm_baseline/<date>_llm-suite_<name>/`):

- Suite-level: `comparison_table.csv`, `comparison_by_error_type.csv`, `manifest.json`
- Per model under `models/<slug>/`:
  - `results/predictions.csv`, `results/eval_summary.txt`
  - `plots/confusion_matrix.png`
  - `logs/output.log` (inference), `logs/metrics.log` (summary)
  - `command.txt`, `command_metrics.txt` (reproducibility)

Qwen3 models get `/no_think` prepended automatically (non-reasoning JSON answers). If the first 50 rows all fail JSON parsing, inference aborts early instead of burning a full eval run.

Standalone `eval_llm_baseline.py` / `llm_eval_metrics.py` runs (without `--output_run_dir`) still create a dated folder under `runs/llm_baseline/` as before.

If `matplotlib` is missing from `Rad_Report_Error_Detection-LLM`, install it once:

```bash
conda activate Rad_Report_Error_Detection-LLM
pip install matplotlib
```
