#!/usr/bin/env bash
# BERT inconsistency detector (binary hard-single) — scaling + pretrain ablation sweep.
# Each experiment is independent: a failure does not stop the rest.
#
# Init variants (INIT_VARIANTS, space-separated) map to pretraining ablation cells:
#   both - MLM+CLS init (default bert_cls_classification_mxln<len>[_big|_8L|_12L])
#   mlm  - MLM only (--wo_cls_init -> bert-pretrained_mxlen_<len>...)
#   cls  - CLS only, no MLM (--init_checkpoint ..._nomlm; run pretraining cls variant first)
#   none - random init (--scratch_init)
#
# Usage (from anywhere):
#   bash scripts/run_bert_sequence.sh
#
# Examples:
#   # Remaining ablation cells (skip combos you already ran):
#   INIT_VARIANTS="mlm cls none" SKIP_COMBOS="big:mlm big:both 8L:both 12L:both" \
#     MODEL_SIZES="big 8L 12L" GPU=1 bash scripts/run_bert_sequence.sh
#
#   # Single cell:
#   INIT_VARIANTS=mlm MODEL_SIZES=8L bash scripts/run_bert_sequence.sh
#
#   # Compare fresh vs pretrained CLS linear head (both/cls variants only):
#   INIT_VARIANTS=both MODEL_SIZES=8L INIT_CLS_HEAD=1 bash scripts/run_bert_sequence.sh
#
# Background:
#   nohup bash scripts/run_bert_sequence.sh \
#     > trained_models/runs/bert_binary/ablation_sweep.log 2>&1 &
#   echo $!

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-Rad_Report_Error_Detection}"
BINARY_SCRIPT="${REPO_ROOT}/src/bert_inconsistency_detector.py"
PRETRAIN_RUNS_DIR="${REPO_ROOT}/trained_models/runs/pretraining"

VAL_CSV="${REPO_ROOT}/data/eval_datasets/val_hard_negative_eval_single_swap_v5.csv"
EVAL_CSV="${REPO_ROOT}/data/eval_datasets/hard_negative_eval_v6_single_swap.csv"

# ---------------------------------------------------------------------------
# Edit these defaults for your sweep
# ---------------------------------------------------------------------------
SEQ_LENGTH="${SEQ_LENGTH:-600}"
# Space-separated: big 8L 12L (aliases 6 8 12 also work)
MODEL_SIZES="${MODEL_SIZES:-big 8L 12L}"
# Space-separated: both mlm cls none
INIT_VARIANTS="${INIT_VARIANTS:-mlm cls none}"
# Space-separated combos to skip, format <model_size>:<variant>  e.g. big:mlm
SKIP_COMBOS="${SKIP_COMBOS-big:mlm big:both 8L:both 12L:both}"
GPU="${GPU:-1}"
# Within-tier severity synonym jitter on uncorrupted positive examples (artifact mitigation).
SEVERITY_JITTER_PROB="${SEVERITY_JITTER_PROB:-0.4}"
# Set to 1 for both/cls variants: load cls_classifier.pt from the init checkpoint.
INIT_CLS_HEAD="${INIT_CLS_HEAD:-}"
# ---------------------------------------------------------------------------

export CUDA_VISIBLE_DEVICES="${GPU}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

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

cls_nomlm_checkpoint_name() {
  local model_size="$1"
  echo "bert_cls_classification_mxln${SEQ_LENGTH}$(checkpoint_suffix "${model_size}")_nomlm"
}

should_skip_combo() {
  local model_size="$1"
  local variant="$2"
  local combo="${model_size}:${variant}"
  [[ " ${SKIP_COMBOS} " == *" ${combo} "* ]]
}

run_one() {
  local label="$1"
  shift
  n=$((n + 1))

  echo ""
  echo "================================================================================"
  echo "Experiment ${n}: ${label}"
  echo "Started: $(date -Iseconds)"
  echo "Command: python -u ${BINARY_SCRIPT} $*"
  echo "================================================================================"

  if python -u "${BINARY_SCRIPT}" "$@"; then
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
BINARY_COMMON_ARGS=(
  --train_mode hard
  --swap_instances single
  --severity_jitter_prob "${SEVERITY_JITTER_PROB}"
  --error_type_weights "${ERROR_WEIGHTS}"
  --epochs 20
)
if [[ -f "${VAL_CSV}" ]]; then
  BINARY_COMMON_ARGS+=(--val_hard_eval_csv "${VAL_CSV}")
else
  echo "Val benchmark CSV not found (${VAL_CSV}); validation uses the val split."
fi
if [[ -f "${EVAL_CSV}" ]]; then
  BINARY_COMMON_ARGS+=(--eval_csv_hard "${EVAL_CSV}")
else
  echo "Post-train benchmark CSV not found (${EVAL_CSV}); skipping that eval."
fi

echo "BERT inconsistency ablation sweep"
echo "  seq_length=${SEQ_LENGTH}"
echo "  model_sizes=${MODEL_SIZES}"
echo "  init_variants=${INIT_VARIANTS}"
echo "  skip_combos=${SKIP_COMBOS:-<none>}"
echo "  severity_jitter_prob=${SEVERITY_JITTER_PROB}"
echo "  init_cls_head=${INIT_CLS_HEAD:-<off>}"
echo "  gpu=${GPU}"

for model_size in ${MODEL_SIZES}; do
  tag="$(size_tag "${model_size}")"
  for variant in ${INIT_VARIANTS}; do
    if should_skip_combo "${model_size}" "${variant}"; then
      echo "Skipping ${model_size} variant=${variant} (listed in SKIP_COMBOS)"
      continue
    fi

    init_args=()
    run_suffix="${variant}"
    case "${variant}" in
      both)
        init_args=()
        run_suffix="mlmcls"
        ;;
      mlm)
        init_args=(--wo_cls_init)
        run_suffix="mlm"
        ;;
      cls)
        nomlm_ckpt="$(cls_nomlm_checkpoint_name "${model_size}")"
        nomlm_model_dir="${PRETRAIN_RUNS_DIR}/${nomlm_ckpt}/model"
        if [[ ! -f "${nomlm_model_dir}/config.json" ]]; then
          echo "WARNING: missing CLS-only checkpoint ${nomlm_model_dir}"
          echo "         Run: PRETRAIN_VARIANTS=cls MODEL_SIZES=${model_size} bash scripts/run_bert_pretraining_sequence.sh"
          fail=$((fail + 1))
          failed_labels+=("missing-cls-nomlm-${model_size}")
          continue
        fi
        init_args=(--init_checkpoint "${nomlm_ckpt}")
        run_suffix="cls"
        ;;
      none)
        init_args=(--scratch_init)
        run_suffix="scratch"
        ;;
      *)
        echo "Unknown INIT_VARIANT: ${variant}" >&2
        exit 1
        ;;
    esac

    cls_head_args=()
    if [[ -n "${INIT_CLS_HEAD}" && "${INIT_CLS_HEAD}" != "0" ]]; then
      case "${variant}" in
        both|cls)
          cls_head_args=(--init_cls_head)
          run_suffix="${run_suffix}-preclshead"
          ;;
        *)
          echo "WARNING: INIT_CLS_HEAD ignored for variant=${variant} (only both/cls supported)"
          ;;
      esac
    fi

    run_one "hard-${run_suffix}-cls${SEQ_LENGTH}-${tag}" \
      "${BINARY_COMMON_ARGS[@]}" \
      "${init_args[@]}" \
      "${cls_head_args[@]}" \
      --seq_length "${SEQ_LENGTH}" \
      --model_size "${model_size}" \
      --run_name "bert_inconsistency-hard-single-${run_suffix}-cls${SEQ_LENGTH}-${tag}"
  done
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
echo "Binary runs: ${REPO_ROOT}/trained_models/runs/bert_binary/"
