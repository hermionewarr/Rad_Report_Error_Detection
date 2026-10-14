# Rad_Report_Error_Detection

Code for paper: [Error Detection for PET/CT Radiology Reports: Domain-Specific vs Large Language Models](https://arxiv.org/pdf/2608.30021)

<img width="750" height="500" alt="Screenshot 2026-10-14 at 11 21 48" src="https://github.com/user-attachments/assets/0ea896da-8bb9-4adc-91bf-5dd2af033076" />


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

A synthetic stand-in is already at `data/dummy/` (`reports.csv` plus `train.csv`,
`val.csv`, `test.csv`) with the same columns as a real reports export. Training
uses that until you point at your own files.

```bash
export REPORTS_CSV=/path/to/your/reports.csv     # Findings, Conclusion, split, plus the other export columns
export TRAIN_CSV=/path/to/train.csv              # optional; else data/train_text_path, else data/dummy/train.csv
export VAL_CSV=/path/to/val.csv
export TEST_CSV=/path/to/test.csv
export TOKENIZER_PATH=/path/to/hf_tokenizer_dir
export CHECKPOINTS_ROOT=/path/to/extra_weights    # optional
```

Or drop files at:

- `data/reports.csv` (used instead of `data/dummy/reports.csv` when present)
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

## Training Domain-specific BERT models

1. run src/bert_model.py to do the MLM pretraining
2. run src/bert_model_cls_classification.py to pretrain the CLS token
3. run src/bert_inconsistency_detector.py (or src/bert_inconsistency_multiclass.py) to fine-tune on the error detection task of interest.

In scripts, there are bash files to run these sequentially.
   
## Imports

```python
from paths import data_path, eval_data_path, DEFAULT_REPORTS_CSV, DEFAULT_TOKENIZER_PATH
from src.predictions_eval import write_eval_artifacts
from bert_architectures import bert_mlm_config
```
