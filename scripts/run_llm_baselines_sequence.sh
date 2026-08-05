#!/usr/bin/env bash
# Run several run_llm_baselines experiments one after another (e.g. prompt 2, 3, 4).
# Each experiment is independent: a failure does not stop the rest.
#
# Usage (from anywhere):
#   bash scripts/run_llm_baselines_sequence.sh
#
# Override defaults via env vars, e.g.:
#   GPU=1 LIMIT=10 bash scripts/run_llm_baselines_sequence.sh
#   PROMPTS="2 4" bash scripts/run_llm_baselines_sequence.sh
#   PROMPTS="3 4 5" RESUME=1 bash scripts/run_llm_baselines_sequence.sh
#
# Eval mode: MODE defaults to hard (full benchmark). Prompts 5 and 6 override
# to severity / lat so single-error-type prompts match the filtered eval subset.
#
# Background:
#   nohup bash scripts/run_llm_baselines_sequence.sh \
#     > trained_models/runs/llm_baseline/sequence_sweep.log 2>&1 &
#   echo $!
#
# Monitor:
#   tail -f trained_models/runs/llm_baseline/sequence_sweep.log

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-Rad_Report_Error_Detection-LLM}"
SCRIPT="${REPO_ROOT}/scripts/run_llm_baselines.py"
RUNS_DIR="${REPO_ROOT}/trained_models/runs/llm_baseline"
LOCK_FILE="${RUNS_DIR}/.sequence_sweep.lock"

# ---------------------------------------------------------------------------
# Edit these defaults for your sweep
# ---------------------------------------------------------------------------
EVAL_CSV="${EVAL_CSV:-${REPO_ROOT}/data/eval_datasets/hard_negative_eval_v6_single_swap.csv}"
MODE="${MODE:-hard}"
MODEL_PREFIX="${MODEL_PREFIX:-}"
GPU="${GPU:-0}"
MANAGE_VLLM="${MANAGE_VLLM:-1}"   # 1 = pass --manage_vllm, 0 = expect server already up
LIMIT="${LIMIT:-}"                # e.g. LIMIT=10 for smoke tests; empty = full eval
SKIP_EXISTING="${SKIP_EXISTING:-0}"  # 1 = pass --skip_existing
RESUME="${RESUME:-0}"              # 1 = pass --resume (continue from predictions.checkpoint.csv)

# Space-separated prompt ids (must exist in llm_eval_prompts.PROMPT_VARIANTS)
#PROMPTS="${PROMPTS:-2 3 4 5}"
#PROMPTS="${PROMPTS:-7 6 5}"
PROMPTS="${PROMPTS:-4}"
# ---------------------------------------------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p "${RUNS_DIR}"
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
  echo "ERROR: Another sequence sweep holds ${LOCK_FILE}. Stop it before restarting."
  exit 1
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# scipy/vLLM wheels need conda's libstdc++ (CXXABI_1.3.15), not the system one.
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

suite_name_for_prompt() {
  local prompt="$1"
  # Keep p4 naming aligned with prior multiclass runs
  if [[ "${prompt}" == "4" ]]; then
    echo "${MODEL_PREFIX}_p4_mc"
  else
    echo "${MODEL_PREFIX}_p${prompt}"
  fi
}

mode_for_prompt() {
  local prompt="$1"
  case "${prompt}" in
    5) echo "severity" ;;
    6) echo "lat" ;;
    *) echo "${MODE}" ;;
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
  echo "Command: python -u ${SCRIPT} $*"
  echo "================================================================================"

  if python -u "${SCRIPT}" "$@"; then
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

echo "LLM baseline prompt sweep"
echo "  conda env:  ${CONDA_ENV}"
echo "  eval csv:   ${EVAL_CSV}"
echo "  mode:       ${MODE} (prompt 5→severity, prompt 6→lat)"
echo "  gpu:        ${GPU}"
echo "  prompts:    ${PROMPTS}"
echo "  limit:      ${LIMIT:-<none>}"
echo "  resume:     ${RESUME}"
echo "  runs dir:   ${RUNS_DIR}"

for prompt in ${PROMPTS}; do
  suite_name="$(suite_name_for_prompt "${prompt}")"
  prompt_mode="$(mode_for_prompt "${prompt}")"
  label="${suite_name}"

  cmd=(
    --eval_csv "${EVAL_CSV}"
    --mode "${prompt_mode}"
    --suite_name "${suite_name}"
    --prompt "${prompt}"
  )

  if [[ "${MANAGE_VLLM}" == "1" ]]; then
    cmd+=(--manage_vllm)
  fi

  if [[ -n "${GPU}" ]]; then
    cmd+=(--gpu "${GPU}")
  fi

  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    cmd+=(--skip_existing)
  fi

  if [[ "${RESUME}" == "1" ]]; then
    cmd+=(--resume)
  fi

  run_one "${label}" "${cmd[@]}"
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
echo "Runs live under: ${RUNS_DIR}/"
