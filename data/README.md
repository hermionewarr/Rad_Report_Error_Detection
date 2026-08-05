# Data

Put your (private) inputs here. Nothing clinical is shipped with this repo.

| Path / env | Role |
|------------|------|
| `reports.csv` or `REPORTS_CSV` | Master reports (columns: `Findings`, `Conclusion`, optional `split`) |
| `train_text_path`, `val_text_path`, `test_text_path` | Split files for training (written by MLM prep or by you) |
| `tokenizer/` or `TOKENIZER_PATH` | HuggingFace tokenizer directory |
| `eval_datasets/*.csv` | Hard-negative / multiclass eval sets (build with `build_eval_datasets.py`) |

See `eval_datasets/README.md` for eval generation.
