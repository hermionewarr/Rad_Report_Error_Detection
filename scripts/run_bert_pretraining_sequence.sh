#!/usr/bin/env bash
# MLM / CLS pretraining for BERT scaling + ablation study (sizes 6 / 8 / 12).
# Each pipeline step is independent: a failure is logged and the sequence continues.
#
# Pretraining variants (PRETRAIN_VARIANTS, space-separated):
#   both  - MLM then CLS (default full pipeline)
#   mlm   - MLM only (fine-tune later with --wo_cls_init)
#   cls   - CLS from random init, saved as bert_cls_classification_mxln<len>_<size>_nomlm
#   none  - skip pretraining (for scratch fine-tuning downstream)
#
# After pretraining, run error-detection fine-tuning with:
#   bash scripts/run_bert_sequence.sh
#
# Usage (from anywhere):
#   bash scripts/run_bert_pretraining_sequence.sh
#
# Examples:
#   # Full MLM+CLS for 6L / 8L / 12L (already done for your main scaling runs):
#   PRETRAIN_VARIANTS=both MODEL_SIZES="big 8L 12L" bash scripts/run_bert_pretraining_sequence.sh
#
#   # Create CLS-only (no MLM) init checkpoints needed for the ablation:
#   PRETRAIN_VARIANTS=cls MODEL_SIZES="big 8L 12L" GPU=1 bash scripts/run_bert_pretraining_sequence.sh
#
#   # MLM-only checkpoints (if missing):
#   PRETRAIN_VARIANTS=mlm MODEL_SIZES="8L 12L" bash scripts/run_bert_pretraining_sequence.sh
#
# Model sizes:
#   small / 4  -> 4 layers, hidden 256
#   big   / 6  -> 6 layers, hidden 384
#   8L    / 8  -> 8 layers, hidden 512
#   12L   / 12 -> 12 layers, hidden 768 (BERT-base scale)
#
# Background:
#   nohup bash scripts/run_bert_pretraining_sequence.sh \
#     > trained_models/runs/pretraining/pretraining_ablation.log 2>&1 &
#   echo $!

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-Rad_Report_Error_Detection}"
MLM_SCRIPT="${REPO_ROOT}/src/bert_model.py"
CLS_SCRIPT="${REPO_ROOT}/src/bert_cls_classification.py"
RUNS_DIR="${REPO_ROOT}/trained_models/runs/pretraining"

# ---------------------------------------------------------------------------
# Edit these defaults for your sweep
# ---------------------------------------------------------------------------
SEQ_LENGTH="${SEQ_LENGTH:-600}"
# Space-separated: big 8L 12L (aliases 6 8 12 also work)
MODEL_SIZES="${MODEL_SIZES:-big 8L 12L}"
# Space-separated: both mlm cls none
PRETRAIN_VARIANTS="${PRETRAIN_VARIANTS:-cls}"
GPU="${GPU:-1}"
MLM_EPOCHS="${MLM_EPOCHS:-50}"
CLS_EPOCHS="${CLS_EPOCHS:-20}"
SKIP_POST_EVAL="${SKIP_POST_EVAL:-1}"  # 1 = skip bert_model.py clustering/fill-mask probes
# Optional per-size batch overrides (empty = script defaults: 32/32/32/16 MLM, 128/128/128/64 CLS)
MLM_BATCH_SMALL="${MLM_BATCH_SMALL:-}"
MLM_BATCH_BIG="${MLM_BATCH_BIG:-}"
MLM_BATCH_8L="${MLM_BATCH_8L:-32}"
MLM_BATCH_12L="${MLM_BATCH_12L:-32}"
CLS_BATCH_SMALL="${CLS_BATCH_SMALL:-}"
CLS_BATCH_BIG="${CLS_BATCH_BIG:-}"
CLS_BATCH_8L="${CLS_BATCH_8L:-}"
CLS_BATCH_12L="${CLS_BATCH_12L:-64}"
# Probe-c MLM hyperparams for deeper models (override per size if needed)
MLM_LR_8L="${MLM_LR_8L:-2e-4}"
MLM_WARMUP_8L="${MLM_WARMUP_8L:-1000}"
MLM_PRECISION_8L="${MLM_PRECISION_8L:-bf16}"
MLM_LR_12L="${MLM_LR_12L:-2e-4}"
MLM_WARMUP_12L="${MLM_WARMUP_12L:-1000}"
MLM_PRECISION_12L="${MLM_PRECISION_12L:-bf16}"
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

checkpoint_suffix() {
  local size="$1"
  case "${size}" in
    small|4|4L|4l) echo "" ;;
    big|6|6L|6l)   echo "_big" ;;
    8L|8|8l|medium) echo "_8L" ;;
    12L|12|12l|xl|base) echo "_12L" ;;
    *) echo "unknown model size: ${size}" >&2; return 1 ;;
  esac
}

cls_nomlm_checkpoint_name() {
  local model_size="$1"
  echo "bert_cls_classification_mxln${SEQ_LENGTH}$(checkpoint_suffix "${model_size}")_nomlm"
}

mlm_batch_for_size() {
  local size="$1"
  case "${size}" in
    small|4|4L|4l) echo "${MLM_BATCH_SMALL:-32}" ;;
    big|6|6L|6l)   echo "${MLM_BATCH_BIG:-32}" ;;
    8L|8|8l|medium) echo "${MLM_BATCH_8L:-32}" ;;
    12L|12|12l|xl|base) echo "${MLM_BATCH_12L:-32}" ;;
    *) echo "unknown model size: ${size}" >&2; return 1 ;;
  esac
}

cls_batch_for_size() {
  local size="$1"
  case "${size}" in
    small|4|4L|4l) echo "${CLS_BATCH_SMALL:-128}" ;;
    big|6|6L|6l)   echo "${CLS_BATCH_BIG:-128}" ;;
    8L|8|8l|medium) echo "${CLS_BATCH_8L:-128}" ;;
    12L|12|12l|xl|base) echo "${CLS_BATCH_12L:-64}" ;;
    *) echo "unknown model size: ${size}" >&2; return 1 ;;
  esac
}

mlm_extra_args_for_size() {
  local size="$1"
  case "${size}" in
    8L|8|8l|medium)
      echo --learning_rate "${MLM_LR_8L}" --warmup_steps "${MLM_WARMUP_8L}" --precision "${MLM_PRECISION_8L}"
      ;;
    12L|12|12l|xl|base)
      echo --learning_rate "${MLM_LR_12L}" --warmup_steps "${MLM_WARMUP_12L}" --precision "${MLM_PRECISION_12L}"
      ;;
    *) ;;
  esac
}

run_one() {
  local label="$1"
  shift
  n=$((n + 1))

  echo ""
  echo "================================================================================"
  echo "Step ${n}: ${label}"
  echo "Started: $(date -Iseconds)"
  echo "Command: python -u $*"
  echo "================================================================================"

  if python -u "$@"; then
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

run_mlm() {
  local model_size="$1"
  local mlm_batch
  mlm_batch="$(mlm_batch_for_size "${model_size}")"

  local mlm_args=(
    "${MLM_SCRIPT}"
    --seq_length "${SEQ_LENGTH}"
    --model_size "${model_size}"
    --epochs "${MLM_EPOCHS}"
    --batch_size "${mlm_batch}"
  )
  # shellcheck disable=SC2206
  mlm_extra=( $(mlm_extra_args_for_size "${model_size}") )
  if ((${#mlm_extra[@]} > 0)); then
    mlm_args+=("${mlm_extra[@]}")
  fi
  if [[ "${SKIP_POST_EVAL}" == "1" ]]; then
    mlm_args+=(--skip_post_eval)
  fi

  run_one "mlm-${model_size}-mxlen${SEQ_LENGTH}" "${mlm_args[@]}"
}

run_cls() {
  local model_size="$1"
  local scratch="${2:-0}"
  local cls_batch ckpt_name output_dir cls_args

  cls_batch="$(cls_batch_for_size "${model_size}")"
  if [[ "${scratch}" == "1" ]]; then
    ckpt_name="$(cls_nomlm_checkpoint_name "${model_size}")"
    output_dir="${RUNS_DIR}/${ckpt_name}/model"
    cls_args=(
      "${CLS_SCRIPT}"
      --seq_length "${SEQ_LENGTH}"
      --model_size "${model_size}"
      --epochs "${CLS_EPOCHS}"
      --batch_size "${cls_batch}"
      --scratch_init
      --run_name "${ckpt_name}"
      --output_model_dir "${output_dir}"
    )
    run_one "cls-nomlm-${model_size}-mxln${SEQ_LENGTH}" "${cls_args[@]}"
  else
    cls_args=(
      "${CLS_SCRIPT}"
      --seq_length "${SEQ_LENGTH}"
      --model_size "${model_size}"
      --epochs "${CLS_EPOCHS}"
      --batch_size "${cls_batch}"
    )
    run_one "cls-${model_size}-mxln${SEQ_LENGTH}" "${cls_args[@]}"
  fi
}

run_pretrain_variant() {
  local variant="$1"
  local model_size="$2"

  case "${variant}" in
    both)
      run_mlm "${model_size}"
      run_cls "${model_size}" 0
      ;;
    mlm)
      run_mlm "${model_size}"
      ;;
    cls)
      run_cls "${model_size}" 1
      ;;
    none)
      echo "Skipping pretraining for variant=none (model_size=${model_size})"
      ;;
    *)
      echo "Unknown PRETRAIN_VARIANT: ${variant}" >&2
      return 1
      ;;
  esac
}

n=0
ok=0
fail=0
failed_labels=()

echo "BERT pretraining ablation"
echo "  seq_length=${SEQ_LENGTH}"
echo "  model_sizes=${MODEL_SIZES}"
echo "  pretrain_variants=${PRETRAIN_VARIANTS}"
echo "  gpu=${GPU}"
echo "  mlm_epochs=${MLM_EPOCHS}  cls_epochs=${CLS_EPOCHS}"
echo "  mlm_batch_sizes: small=${MLM_BATCH_SMALL:-32} big=${MLM_BATCH_BIG:-32} 8L=${MLM_BATCH_8L:-32} 12L=${MLM_BATCH_12L:-32}"
echo "  cls_batch_sizes: small=${CLS_BATCH_SMALL:-128} big=${CLS_BATCH_BIG:-128} 8L=${CLS_BATCH_8L:-128} 12L=${CLS_BATCH_12L:-64}"
echo "  mlm_8L:  lr=${MLM_LR_8L} warmup=${MLM_WARMUP_8L} precision=${MLM_PRECISION_8L}"
echo "  mlm_12L: lr=${MLM_LR_12L} warmup=${MLM_WARMUP_12L} precision=${MLM_PRECISION_12L}"
echo ""
echo "Ablation matrix (error-detection init after pretraining):"
echo "  both -> bert_cls_classification_mxln<len>[_big|_8L|_12L]"
echo "  mlm  -> bert-pretrained_mxlen_<len>[_big|_8L|_12L]  (fine-tune with --wo_cls_init)"
echo "  cls  -> bert_cls_classification_mxln<len>[_suffix]_nomlm"
echo "  none -> --scratch_init at fine-tune time (run_bert_sequence.sh)"

for variant in ${PRETRAIN_VARIANTS}; do
  for model_size in ${MODEL_SIZES}; do
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "Pretrain variant=${variant}  model_size=${model_size}"
    echo "--------------------------------------------------------------------------------"
    run_pretrain_variant "${variant}" "${model_size}"
  done
done

echo ""
echo "================================================================================"
echo "PRETRAINING SEQUENCE SUMMARY"
echo "================================================================================"
echo "Total steps: ${n}"
echo "OK:          ${ok}"
echo "Failed:      ${fail}"
if ((${#failed_labels[@]} > 0)); then
  echo "Failed steps:"
  for lbl in "${failed_labels[@]}"; do
    echo "  - ${lbl}"
  done
fi
echo "Finished: $(date -Iseconds)"
echo "Checkpoints under: ${RUNS_DIR}/"
echo "Next: bash scripts/run_bert_sequence.sh"
