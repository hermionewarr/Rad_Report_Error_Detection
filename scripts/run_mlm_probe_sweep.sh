#!/usr/bin/env bash
# Short MLM hyperparameter probe sweep for deeper BERT sizes (8L / 12L).
# Runs three recipes × 5 epochs each (LR scheduler assumes 50 epochs for fair comparison).
#
# Probes:
#   A  baseline   lr=5e-4  warmup=0     fp16  (current recipe)
#   B  bert-like  lr=1e-4  warmup=1000  fp16
#   C  stable     lr=2e-4  warmup=1000  bf16
#   D  b-fixed    lr=1e-4  warmup=1000  bf16  (b with warmup + bf16)
#   E  a-fixed    lr=5e-4  warmup=1000  bf16  (a with warmup + bf16)
#
# Usage (from anywhere):
#   bash scripts/run_mlm_probe_sweep.sh
#
# Overrides:
#   MODEL_SIZE=8L SEQ_LENGTH=600 GPU=1 EPOCHS=5 bash scripts/run_mlm_probe_sweep.sh
#   PROBES="a b" MODEL_SIZE=12L bash scripts/run_mlm_probe_sweep.sh
#
# Background:
#   nohup bash scripts/run_mlm_probe_sweep.sh \
#     > trained_models/runs/pretraining/mlm_probe_sweep.log 2>&1 &
#   echo $!
#
# Gate (manual): after ~1500 steps, eval_loss should be < 4.0 if learning like 6L big.
# Stuck near ~5.7 means the recipe failed.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-PTCT3}"
MLM_SCRIPT="${REPO_ROOT}/src/bert_model.py"
RUNS_DIR="${REPO_ROOT}/trained_models/runs/pretraining"

SEQ_LENGTH="${SEQ_LENGTH:-600}"
MODEL_SIZE="${MODEL_SIZE:-8L}"
GPU="${GPU:-1}"
EPOCHS="${EPOCHS:-5}"
# LR scheduler horizon for fair comparison with full 50-epoch pretraining.
SCHEDULER_TOTAL_EPOCHS="${SCHEDULER_TOTAL_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Space-separated subset: a b c d
PROBES="${PROBES:-a b c}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

SWEEP_DIR="${RUNS_DIR}/mlm_probe_sweep_${MODEL_SIZE}_mxlen${SEQ_LENGTH}"
mkdir -p "${SWEEP_DIR}"

n=0
ok=0
fail=0
failed_labels=()

run_probe() {
  local probe_id="$1"
  local label="$2"
  local learning_rate="$3"
  local warmup_steps="$4"
  local precision="$5"

  local out_dir="${SWEEP_DIR}/probe_${probe_id}/model"
  mkdir -p "${out_dir}"

  n=$((n + 1))
  echo ""
  echo "================================================================================"
  echo "Step ${n}: probe-${probe_id} (${label})"
  echo "Started: $(date -Iseconds)"
  echo "  model_size=${MODEL_SIZE}  seq_length=${SEQ_LENGTH}  epochs=${EPOCHS}"
  echo "  scheduler_total_epochs=${SCHEDULER_TOTAL_EPOCHS}  (LR decay matches full pretrain)"
  echo "  learning_rate=${learning_rate}  warmup_steps=${warmup_steps}  precision=${precision}"
  echo "  output_dir=${out_dir}"
  echo "================================================================================"

  if python -u "${MLM_SCRIPT}" \
    --seq_length "${SEQ_LENGTH}" \
    --model_size "${MODEL_SIZE}" \
    --epochs "${EPOCHS}" \
    --scheduler_total_epochs "${SCHEDULER_TOTAL_EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate "${learning_rate}" \
    --warmup_steps "${warmup_steps}" \
    --precision "${precision}" \
    --run_name "mlm_probe_${probe_id}_${MODEL_SIZE}_mxlen${SEQ_LENGTH}" \
    --output_model_dir "${out_dir}" \
    --skip_post_eval; then
    echo "✓ probe-${probe_id} finished OK ($(date -Iseconds))"
    ok=$((ok + 1))
    return 0
  fi

  local code=$?
  echo "✗ probe-${probe_id} failed with exit code ${code} ($(date -Iseconds))"
  fail=$((fail + 1))
  failed_labels+=("probe-${probe_id}")
  return "${code}"
}

probe_enabled() {
  local probe_id="$1"
  local p
  for p in ${PROBES}; do
    if [[ "${p}" == "${probe_id}" ]]; then
      return 0
    fi
  done
  return 1
}

echo "MLM probe sweep"
echo "  sweep_dir=${SWEEP_DIR}"
echo "  model_size=${MODEL_SIZE}  seq_length=${SEQ_LENGTH}  gpu=${GPU}"
echo "  epochs=${EPOCHS}  scheduler_total_epochs=${SCHEDULER_TOTAL_EPOCHS}  batch_size=${BATCH_SIZE}"
echo "  probes=${PROBES}"

if probe_enabled a; then
  run_probe a "baseline" 5e-4 0 fp16 || true
fi
if probe_enabled b; then
  run_probe b "bert-like" 1e-4 1000 fp16 || true
fi
if probe_enabled c; then
  run_probe c "stable-bf16" 2e-4 1000 bf16 || true
fi
if probe_enabled d; then
  run_probe d "b-warmup-bf16" 1e-4 1000 bf16 || true
fi
if probe_enabled e; then
  run_probe e "a-warmup-bf16" 5e-4 1000 bf16 || true
fi

echo ""
echo "================================================================================"
echo "PROBE SWEEP SUMMARY"
echo "================================================================================"
echo "Total: ${n}  OK: ${ok}  Failed: ${fail}"
if ((${#failed_labels[@]} > 0)); then
  echo "Failed:"
  for lbl in "${failed_labels[@]}"; do
    echo "  - ${lbl}"
  done
fi
echo ""
echo "Loss summary (latest checkpoint per probe):"
python - "${SWEEP_DIR}" <<'PY'
import json
import sys
from pathlib import Path

sweep_dir = Path(sys.argv[1])
rows = []
for probe_dir in sorted(sweep_dir.glob("probe_*/model")):
    probe_id = probe_dir.parent.name.replace("probe_", "")
    states = sorted(probe_dir.glob("checkpoint-*/trainer_state.json"))
    if not states:
        print(f"  {probe_id}: no checkpoint trainer_state.json")
        continue
    state = json.loads(states[-1].read_text())
    hist = state.get("log_history", [])
    train_loss = next(
        (e["loss"] for e in reversed(hist) if "loss" in e and "eval_loss" not in e),
        None,
    )
    eval_loss = next((e["eval_loss"] for e in reversed(hist) if "eval_loss" in e), None)
    step = state.get("global_step")
    epoch = state.get("epoch")
    rows.append((probe_id, step, epoch, train_loss, eval_loss))

if not rows:
    print("  (no probe metrics found)")
else:
    print(f"  {'probe':<6} {'step':>6} {'epoch':>6} {'train_loss':>11} {'eval_loss':>11}")
    for probe_id, step, epoch, train_loss, eval_loss in rows:
        tl = f"{train_loss:.4f}" if train_loss is not None else "n/a"
        el = f"{eval_loss:.4f}" if eval_loss is not None else "n/a"
        ep = f"{epoch:.2f}" if epoch is not None else "n/a"
        print(f"  {probe_id:<6} {step:>6} {ep:>6} {tl:>11} {el:>11}")
    print()
    print("  Gate: eval_loss < 4.0 by ~step 1500 suggests real learning (6L big passes).")
    print("  Stuck near ~5.7 = same failure mode as the stalled 8L/12L runs.")
PY

echo ""
echo "Finished: $(date -Iseconds)"
echo "Checkpoints: ${SWEEP_DIR}/probe_<a|b|c|d>/model"
