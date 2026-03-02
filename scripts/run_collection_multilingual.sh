#!/usr/bin/env bash
# ==============================================================================
# OpenAct multilingual collection: gsm8k, mgsm, belebele, theoremqa, mmlu, commonsenseqa
# ==============================================================================
#
# Usage (run from repo root; suitable for tmux):
#   chmod +x scripts/run_collection_multilingual.sh
#   ./scripts/run_collection_multilingual.sh [qwen|llama|mistral|all] [--smoke]
#
# Datasets:
#   - gsm8k         English math (single run)
#   - math          MATH benchmark (single run)
#   - mgsm          Multilingual math; one run per language -> mgsm_en, mgsm_zh, ...
#   - belebele      Multilingual reading; one run per language -> belebele_en, ...
#   - theoremqa     English theorem QA (single run)
#   - mmlu          English MMLU (single run)
#   - commonsenseqa English commonsense QA (single run)
#
# All languages below are implemented in openact-collect (MGSMTask, BelebeleTask).
# Runs with _SUCCESS are skipped (resume-safe). Three models run sequentially for "all".
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

# ------------------------------------------------------------------------------
# Configuration (edit as needed)
# ------------------------------------------------------------------------------
BASE_DIR="${BASE_DIR:-$(pwd)/runs}"
OPENACT_CMD="${OPENACT_CMD:-openact collect}"

MODEL_QWEN="Qwen/Qwen2-7B-Instruct"
MODEL_LLAMA="meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_MISTRAL="mistralai/Mistral-7B-Instruct-v0.3"

MAX_TOKENS=2048
TEMPERATURE=0.0
SEED=42
QUEUE_SIZE=4
HIDDEN_DTYPE="float16"
LAYERS="${LAYERS:-}"

# MGSM: implemented in code as MGSM_LANGUAGES = bn de en es fr ja ru sw te th zh
MGSM_LANGS="${MGSM_LANGS:-en zh ja de fr es ru}"

# Belebele: implemented in code as BELEBELE_LANG_MAP keys = en zh ja de fr es ru ar hi ko pt it th vi bn sw te tr
BELEBELE_LANGS="${BELEBELE_LANGS:-en zh ja de fr es ru ar}"

# ------------------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------------------
MODE="${1:-}"
SMOKE=""
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE="--max-samples 20"
    MODE="${2:-all}"
elif [[ "${2:-}" == "--smoke" ]]; then
    SMOKE="--max-samples 20"
fi
MODE="${MODE:-all}"

EXTRA_LAYERS=""
[[ -n "$LAYERS" ]] && EXTRA_LAYERS="--layers $LAYERS"

mkdir -p "$BASE_DIR"

# ------------------------------------------------------------------------------
# Run one (model, task, output subdir, extra CLI args)
# ------------------------------------------------------------------------------
run_one() {
    local model=$1
    local short=$2
    local task=$3
    local out_subdir=$4
    local extra=${5:-}
    local out="${BASE_DIR}/${short}/${out_subdir}"

    if [[ -f "${out}/_SUCCESS" ]]; then
        echo "  [skip] ${short}/${out_subdir} (_SUCCESS present)"
        return 0
    fi

    echo "  [run] ${short}/${out_subdir}"
    $OPENACT_CMD \
        --model "$model" \
        --task "$task" \
        --output "$out" \
        --max-tokens "$MAX_TOKENS" \
        --temperature "$TEMPERATURE" \
        --seed "$SEED" \
        --queue-size "$QUEUE_SIZE" \
        --hidden-dtype "$HIDDEN_DTYPE" \
        $SMOKE \
        $EXTRA_LAYERS \
        $extra \
    || { echo "  [FAIL] ${short}/${out_subdir}"; return 1; }
}

# ------------------------------------------------------------------------------
# One model: gsm8k -> math -> mgsm (per lang) -> belebele (per lang) -> theoremqa -> mmlu -> commonsenseqa
# ------------------------------------------------------------------------------
collect_model() {
    local model=$1
    local short=$2
    echo ""
    echo "========== $short =========="

    run_one "$model" "$short" "gsm8k" "gsm8k" ""
    run_one "$model" "$short" "math" "math" ""

    for lang in $MGSM_LANGS; do
        run_one "$model" "$short" "mgsm" "mgsm_${lang}" "--language ${lang}"
    done

    for lang in $BELEBELE_LANGS; do
        run_one "$model" "$short" "belebele" "belebele_${lang}" "--language ${lang}"
    done

    run_one "$model" "$short" "theoremqa" "theoremqa" ""
    run_one "$model" "$short" "mmlu" "mmlu" ""
    run_one "$model" "$short" "commonsenseqa" "commonsenseqa" ""

    echo "========== $short done =========="
}

# ------------------------------------------------------------------------------
# Entrypoint: run one model or all three sequentially
# ------------------------------------------------------------------------------
echo "OpenAct multilingual collection (gsm8k, math, mgsm, belebele, theoremqa, mmlu, commonsenseqa)"
echo "  BASE_DIR=$BASE_DIR"
echo "  MODE=$MODE  SMOKE=$SMOKE"
echo "  MGSM_LANGS=$MGSM_LANGS"
echo "  BELEBELE_LANGS=$BELEBELE_LANGS"
echo ""

case "$MODE" in
    qwen)
        collect_model "$MODEL_QWEN" "Qwen2-7B-Instruct"
        ;;
    llama)
        collect_model "$MODEL_LLAMA" "Meta-Llama-3-8B-Instruct"
        ;;
    mistral)
        collect_model "$MODEL_MISTRAL" "Mistral-7B-Instruct-v03"
        ;;
    all)
        collect_model "$MODEL_QWEN" "Qwen2-7B-Instruct"
        collect_model "$MODEL_LLAMA" "Meta-Llama-3-8B-Instruct"
        collect_model "$MODEL_MISTRAL" "Mistral-7B-Instruct-v03"
        ;;
    *)
        echo "Usage: $0 {qwen|llama|mistral|all} [--smoke]"
        echo ""
        echo "  qwen|llama|mistral  Run one model: gsm8k + math + mgsm (per lang) + belebele (per lang) + theoremqa + mmlu + commonsenseqa"
        echo "  all                 Run all three models sequentially (Qwen -> Llama -> Mistral)"
        echo "  --smoke             Limit to 20 samples per task (quick test)"
        echo ""
        echo "Override languages: MGSM_LANGS='en zh' BELEBELE_LANGS='en zh' $0 all"
        exit 0
        ;;
esac

echo ""
echo "Done. Completed runs: find $BASE_DIR -name _SUCCESS | wc -l"
exit 0
