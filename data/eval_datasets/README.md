# Evaluation CSVs (not shipped)

Generate from your split files, e.g.:

```bash
python data/build_eval_datasets.py \
  --mode hard_v5 \
  --swap_instances single \
  --input data/test_text_path \
  --output data/eval_datasets/hard_negative_eval.csv
```

Or place pre-built eval CSVs in this folder.
