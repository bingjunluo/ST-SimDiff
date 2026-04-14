#!/bin/bash
# ST-SimDiff VideoMME Evaluation Script
# Usage: bash run_eval.sh [--limit N]
#
# Environment variables (override defaults):
#   MODEL_PATH   - path to LLaVA-Video model (default: ../model/llava-video)
#   HF_ENDPOINT  - HuggingFace endpoint (default: https://huggingface.co)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

# Parse arguments
LIMIT_ARG=""
if [ "$1" = "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
fi

# Model path
MODEL_PATH="${MODEL_PATH:-../model/llava-video}"

# Evaluation parameters
COST=0.3
EVENT_UPPER_BOUND=0.2
SIMILARITY_LOWER_BOUND=0.8
TASK=videomme

echo "=== ST-SimDiff VideoMME Evaluation ==="
echo "Model: ${MODEL_PATH}"
echo "Parameters: cost=${COST}, event_upper_bound=${EVENT_UPPER_BOUND}, similarity_lower_bound=${SIMILARITY_LOWER_BOUND}"
echo "Task: ${TASK}"
if [ -n "${LIMIT_ARG}" ]; then
    echo "Limit: $2 samples"
fi

cd "${SCRIPT_DIR}"

python -m accelerate.commands.launch \
    --num_processes=1 \
    -m lmms_eval \
    --model llava_video \
    --model_args "pretrained=${MODEL_PATH},conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=64,cost=${COST},similarity_lower_bound=${SIMILARITY_LOWER_BOUND},event_upper_bound=${EVENT_UPPER_BOUND},merge_type=new_topk,right=True,bottom=True,spatial=True,temporal=True,strategy=3,mm_spatial_pool_mode=bilinear" \
    --tasks ${TASK} \
    --batch_size 1 \
    --output_path ./logs/ \
    ${LIMIT_ARG}
