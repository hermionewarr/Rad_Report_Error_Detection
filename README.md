# Rad_Report_Error_Detection

Standalone code for findings–conclusion inconsistency detection in radiology
reports (BERT + local LLM baselines). **Method / supplementary code only** —
no clinical data, figures, or trained weights.

## Layout

```
paths.py                  # all path defaults (data/, runs/, env overrides)
bert_architectures.py
report_split_io.py
src/                      # train + eval
scripts/                  # sweeps + figure generators
data/                     # put your reports / tokenizer / eval CSVs here
config/                   # LLM model list JSON
no_findings_analysis/     # NF labelling helpers
trained_models/           # local checkpoints live here when you train
```

## Setup

```bash
cd /path/to/Rad_Report_Error_Detection
# BERT train / eval / plots (PTCT3-style stack):
conda env create -f environment.yml && conda activate Rad_Report_Error_Detection
# Local LLM baselines (vLLM):
# conda env create -f environment-llm-eval.yml && conda activate Rad_Report_Error_Detection-LLM
export PYTHONPATH=$PWD
```

## Point at your data

Everything resolves under this repo’s `data/` by default. Override with env vars
(defined in `paths.py`):

```bash
export REPORTS_CSV=/path/to/your/reports.csv     # Findings, Conclusion, optional split
export TOKENIZER_PATH=/path/to/hf_tokenizer_dir
export CHECKPOINTS_ROOT=/path/to/extra_weights    # optional
```

Or drop files at:

- `data/reports.csv`
- `data/tokenizer/`
- `data/train_text_path`, `data/val_text_path`, `data/test_text_path`
- `data/eval_datasets/*.csv`

## Run (examples)

```bash
python data/build_eval_datasets.py --help
python src/bert_inconsistency_detector.py --help
python scripts/run_llm_baselines.py --help
```

LLM notes: [`LLM_EVAL_SETUP.md`](LLM_EVAL_SETUP.md).

## Imports

```python
from paths import data_path, eval_data_path, DEFAULT_REPORTS_CSV, DEFAULT_TOKENIZER_PATH
from src.predictions_eval import write_eval_artifacts
from bert_architectures import bert_mlm_config
```
