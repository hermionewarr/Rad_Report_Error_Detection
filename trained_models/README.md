# Trained models / run artifacts

```
trained_models/runs/
  pretraining/
  bert_binary/
  bert_multiclass/
  llm_baseline/
```

Each run typically contains `model/`, `results/`, `plots/`, `logs/`.

**Not shipped by default** (size + licensing): model weights and full prediction dumps.
Point scripts at local checkpoints, or recreate on your own dataset.
