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

- NVIDIA driver + GPU (you have RTX 6000 / A6000 — suitable for 70B with AWQ)
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

`data/eval_datasets/hard_negative_eval_v4_single_swap.csv`

Use **`--mode hard`** for the full v4 hard-negative suite (laterality, site, severity, NF vs disease, **cross_pair_mismatch**, plus consistent). On v4 CSVs, `hard` and `all` load the same rows; use `cross_pair_full` or `lat` for ablations. LLM reports use **`eval_summary.txt`** (overall + per error type), not incidence buckets.

```bash
cd /path/to/Rad_Report_Error_Detection
conda activate Rad_Report_Error_Detection-LLM

python scripts/run_llm_baselines.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v4_single_swap.csv \
  --mode hard \
  --suite_name single-swap-hard-v1 \
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

For a fair BERT comparison, run BERT on the **same** single-swap CSV first, then pass that predictions file:

```bash
--bert_predictions trained_models/runs/<bert-single-swap>/results/predictions.csv \
--bert_label bert-single-swap-v1
```

## 4. Run all models + comparison table (legacy lat example)

**Option A — runner manages vLLM** (loads one model at a time):

```bash
cd /path/to/Rad_Report_Error_Detection
conda activate Rad_Report_Error_Detection-LLM

python scripts/run_llm_baselines.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v3.csv \
  --mode lat \
  --suite_name hard-v3-lat \
  --manage_vllm \
  --bert_predictions trained_models/runs/20260526_my-eval-with-incidence/results/predictions.csv
```

**Option B — you start vLLM manually**, runner only calls eval API:

```bash
# Terminal 1
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --quantization awq \
  --host 127.0.0.1 --port 8000

# Terminal 2 (after each model swap, restart server with next weights)
python scripts/run_llm_baselines.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v3.csv \
  --mode lat \
  --suite_name hard-v3-lat \
  --no_manage_vllm
```

### Outputs

Under `trained_models/runs/<date>_llm-suite_<suite_name>/`:

- `models/<slug>/results/predictions.csv`
- `models/<slug>/results/eval_summary.txt`
- `models/<slug>/plots/confusion_matrix.png`
- `comparison_table.csv` — for spreadsheets / paper
- `comparison_table.md` — markdown table
- `manifest.json` — reproducibility metadata

### Smoke test (10 rows)

Models that fail (e.g. Llama still gated on HF) are **skipped by default**; the suite continues and records them in `manifest.json` under `skipped`.

Use a free GPU so you do not disturb other jobs (example: GPU 1 only):

```bash
python scripts/run_llm_baselines.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v3.csv \
  --mode lat \
  --suite_name smoke \
  --manage_vllm \
  --gpu 1 \
  --limit 10 \
  --models_json config/llm_models_default.json
```

`--gpu 1` sets `CUDA_VISIBLE_DEVICES=1` for the vLLM process only (eval client stays on CPU).

To stop on first failure instead: `--no-skip-on-failure`.

To skip a model without trying it, set `"enabled": false` in `llm_models_default.json` (Llama is disabled there until you approve access).

## 5. Single-model eval (manual)

```bash
python src/eval_llm_baseline.py \
  --eval_csv data/eval_datasets/hard_negative_eval_v3.csv \
  --mode lat \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --run_name llm-lat-llama33

python src/llm_eval_metrics.py \
  --predictions_csv trained_models/runs/<run>/results/predictions.csv \
  --mode lat
```

## Troubleshooting

### Model loads then crashes: `Could not find nvcc`

Weights load (~39GB) then vLLM fails during warmup with `cuda_home='/usr/local/cuda' doesn't exist`.

**Fix (default in runner):** the suite sets `VLLM_USE_FLASHINFER_SAMPLER=0` so sampling does not JIT-compile CUDA kernels. Re-run the smoke test; no system CUDA toolkit required.

**Optional (faster sampling):** install `nvcc` in the conda env, then use `--use_flashinfer_sampler`:

```bash
conda activate Rad_Report_Error_Detection-LLM
conda install -c nvidia cuda-nvcc
export CUDA_HOME="$CONDA_PREFIX"
```

### OOM after model load on a 49GB GPU

Lower context in `llm_models_default.json` (`max_model_len`: 4096) and `"enforce_eager": true` for 72B AWQ (already set for Qwen).

## Paper wording (example)

> LLM baselines used open-weight instruct models served locally via vLLM on institutional hardware. Patient report text was not transmitted to external inference APIs.
