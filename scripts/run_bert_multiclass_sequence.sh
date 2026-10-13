#!/usr/bin/env bash
# BERT multiclass inconsistency detector — hard-single training sweep.
# Aligned with run_bert_sequence.sh training args (jitter, error weights, eval CSVs).
# Each experiment is independent: a failure does not stop the rest.
#
# Usage (from anywhere):
#   bash scripts/run_bert_multiclass_sequence.sh
#
# Examples:
#   MODEL_SIZES="8L 12L" GPU=1 bash scripts/run_bert_multiclass_sequence.sh
#   MODEL_SIZES=big GPU=0 bash scripts/run_bert_multiclass_sequence.sh
#
# Background:
#   nohup bash scripts/run_bert_multiclass_sequence.sh \
#     > trained_models/runs/bert_multiclass/mc_sweep.log 2>&1 &
#   echo $!

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-Rad_Report_Error_Detection}"
MC_SCRIPT="${REPO_ROOT}/src/bert_inconsistency_multiclass.py"

VAL_CSV="${REPO_ROOT}/data/eval_datasets/val_hard_negative_eval_single_swap_v5.csv"
EVAL_CSV="${REPO_ROOT}/data/eval_datasets/hard_negative_eval_v6_single_swap.csv"

# ---------------------------------------------------------------------------
# Edit these defaults for your sweep
# ---------------------------------------------------------------------------
SEQ_LENGTH="${SEQ_LENGTH:-600}"
# Space-separated: small big 8L 12L (aliases 4 6 8 12 also work)
MODEL_SIZES="${MODEL_SIZES:-small big 8L 12L}"
GPU="${GPU:-1}"
SEVERITY_JITTER_PROB="${SEVERITY_JITTER_PROB:-0.4}"
# ---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${GPU}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

size_tag() {
  local size="$1"
  case "${size}" in
    small|4|4L|4l) echo "4" ;;
    big|6|6L|6l)   echo "6" ;;
    8L|8|8l|medium) echo "8L" ;;
    12L|12|12l|xl|base) echo "12L" ;;
    *) echo "${size}" ;;
  esac
}

run_one() {
  local label="$1"
  shift
  n=$((n + 1))

  echo ""
  echo "================================================================================"
  echo "Experiment ${n}: ${label}"
  echo "Started: $(date -Iseconds)"
  echo "Command: python -u ${MC_SCRIPT} $*"
  echo "================================================================================"

  if python -u "${MC_SCRIPT}" "$@"; then
    echo "✓ ${label} finished OK ($(date -Iseconds))"
    ok=$((ok + 1))
    return 0
  else
    local code=$?
    echo "✗ ${label} failed with exit code ${code} ($(date -Iseconds))"
    fail=$((fail + 1))
    failed_labels+=("${label}")
    return "${code}"
  fi
}

n=0
ok=0
fail=0
failed_labels=()

ERROR_WEIGHTS='{"laterality_swap": 3.0, "site_swap": 3.0, "severity_flip": 3.0}'
MC_COMMON_ARGS=(
  --train_mode hard
  --swap_instances single
  --severity_jitter_prob "${SEVERITY_JITTER_PROB}"
  --error_type_weights "${ERROR_WEIGHTS}"
  --mode hard
)
if [[ -f "${VAL_CSV}" ]]; then
  MC_COMMON_ARGS+=(--val_hard_eval_csv "${VAL_CSV}")
else
  echo "Val benchmark CSV not found (${VAL_CSV}); validation uses the val split."
fi
if [[ -f "${EVAL_CSV}" ]]; then
  MC_COMMON_ARGS+=(--eval_csv "${EVAL_CSV}")
else
  echo "Post-train benchmark CSV not found (${EVAL_CSV}); skipping that eval."
fi

echo "BERT multiclass hard-single sweep"
echo "  seq_length=${SEQ_LENGTH}"
echo "  model_sizes=${MODEL_SIZES}"
echo "  severity_jitter_prob=${SEVERITY_JITTER_PROB}"
echo "  error_type_weights=${ERROR_WEIGHTS}"
echo "  val_csv=${VAL_CSV}"
echo "  eval_csv=${EVAL_CSV}"
echo "  gpu=${GPU}"

for model_size in ${MODEL_SIZES}; do
  tag="$(size_tag "${model_size}")"
  run_one "mc-hard-single-cls${SEQ_LENGTH}-${tag}" \
    "${MC_COMMON_ARGS[@]}" \
    --seq_length "${SEQ_LENGTH}" \
    --model_size "${model_size}" \
    --run_name "bert_inconsistency-mc-hard-single-cls${SEQ_LENGTH}-${tag}"
done

echo ""
echo "================================================================================"
echo "SEQUENCE SUMMARY"
echo "================================================================================"
echo "Total:  ${n}"
echo "OK:     ${ok}"
echo "Failed: ${fail}"
if ((${#failed_labels[@]} > 0)); then
  echo "Failed runs:"
  for lbl in "${failed_labels[@]}"; do
    echo "  - ${lbl}"
  done
fi
echo "Finished: $(date -Iseconds)"
echo "Multiclass runs: ${REPO_ROOT}/trained_models/runs/bert_multiclass/"
