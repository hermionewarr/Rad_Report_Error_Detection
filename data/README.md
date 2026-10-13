# Data

Put your (private) inputs here. Nothing clinical is shipped with this repo.

| Path / env | Role |
|------------|------|
| `dummy/reports.csv` (default) or `reports.csv` / `REPORTS_CSV` | Master reports. Dummy rows use the real export columns |
| `dummy/{train,val,test}.csv` or `*_text_path` / `TRAIN_CSV` `VAL_CSV` `TEST_CSV` | Split files for CLS, contrastive, and inconsistency training |
| `tokenizer/` or `TOKENIZER_PATH` | HuggingFace tokenizer directory |
| `eval_datasets/*.csv` | Hard-negative / multiclass eval sets (build with `build_eval_datasets.py`) |

See `eval_datasets/README.md` for eval generation.
