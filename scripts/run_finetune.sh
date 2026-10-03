#!/usr/bin/env bash
# Domain-adaptation run: train adapter -> merge -> GGUF (Q5_K_M) -> Ollama -> eval.
# Reproduces one row of the paper's domain-adaptation table.
#
#   CONFIG_NAME=dora_r16 LORA_R=16 USE_DORA=1 LLAMA_CPP=~/llama.cpp scripts/run_finetune.sh
#
# Needs: a running `ollama serve`, a built llama.cpp checkout (convert_hf_to_gguf.py
# and build/bin/llama-quantize), and requirements-finetune.txt installed.
set -euo pipefail
: "${CONFIG_NAME:?e.g. qlora_r8, qlora_r16, dora_r8, dora_r16}"
: "${LORA_R:?8 or 16}"
: "${USE_DORA:?1 or 0}"
: "${LLAMA_CPP:?path to a built llama.cpp checkout}"
PY=${PYTHON:-python}
BASE_TAG=${BASE_TAG:-qwen2.5-coder:14b}

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT_DIR="$ROOT/models/$CONFIG_NAME"
DORA_FLAG=""
[ "$USE_DORA" = "0" ] && DORA_FLAG="--no-dora"

# Always restore the base coder model, even if a step fails.
trap '$PY -c "from agents.model_router import swap_model; swap_model(\"p4_coder\", \"$BASE_TAG\")" || true' EXIT

echo "--- [1/5] Training ($CONFIG_NAME) ---"
$PY experiments/finetune_qlora.py --data data/verified_corpus.jsonl --output "$OUT_DIR" \
    --epochs 3 --batch-size 4 --lora-r "$LORA_R" $DORA_FLAG

echo "--- [2/5] Merging adapter ---"
$PY experiments/merge_adapter.py --base Qwen/Qwen2.5-Coder-14B-Instruct \
    --adapter "$OUT_DIR/final" --output "$OUT_DIR/merged"

echo "--- [3/5] GGUF conversion + Q5_K_M quantization ---"
$PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$OUT_DIR/merged" --outfile "$OUT_DIR/merged-f16.gguf"
"$LLAMA_CPP/build/bin/llama-quantize" --token-embedding-type q8_0 --output-tensor-type q8_0 \
    "$OUT_DIR/merged-f16.gguf" "$OUT_DIR/merged-q5km.gguf" Q5_K_M
rm -f "$OUT_DIR/merged-f16.gguf"

echo "--- [4/5] Importing into Ollama ---"
MODEL_NAME="p4harness-p4-$CONFIG_NAME"
cat > "$OUT_DIR/Modelfile" <<MODELFILE
FROM $OUT_DIR/merged-q5km.gguf
SYSTEM "You are a P4_16 programmer for bmv2 Service Function Chaining. Output ONLY the modified table and action definitions in a \`\`\`p4 block. Keep under 40 lines. Follow NSH SFC rules: decrement SI, use NoAction() as default, snake_case field names."
MODELFILE
ollama create "$MODEL_NAME" -f "$OUT_DIR/Modelfile"

echo "--- [5/5] Evaluating on the 50-task frozen eval set ---"
$PY -c "from agents.model_router import swap_model; swap_model('p4_coder', '$MODEL_NAME')"
$PY experiments/run_baseline_eval.py --use-moa --parallel 4 \
    --checkpoint "experiments/results/finetune_eval_checkpoint_${CONFIG_NAME}.json"
