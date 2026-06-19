#!/bin/bash
# Usage: bash scripts/eval/eval_sqa3d.sh <pooling> <voxel_size> <target_tokens>
# Example (baseline):  bash scripts/eval/eval_sqa3d.sh native
# Example (3DZip):     bash scripts/eval/eval_sqa3d.sh 3dzip 0.2 64

POOLING=${1:-native}
VOXEL_SIZE=${2:-0.2}
TARGET_TOKENS=${3:-0}

# native = baseline 
if [ "${POOLING}" == "native" ]; then
    POOLING=voxelize
    VOXEL_SIZE=0.2
    TARGET_TOKENS=0
fi

export POOLING
export VOXEL_SIZE
export ADAPTIVE_RATIO=${TARGET_TOKENS}
export MERGE_CUTOFF=${MERGE_CUTOFF:-5}

mkdir -p pred results

PRED_FILE=./pred/sqa3d_${POOLING}_vs${VOXEL_SIZE}_t${TARGET_TOKENS}.json
GT_FILE=playground/data/annotations/llava3d_sqa3d_test_answer.json
QUESTION_FILE=playground/data/annotations/llava-3d-sqa3d_test_question.json

echo "=== SQA3D: POOLING=${POOLING}, VOXEL_SIZE=${VOXEL_SIZE}, TARGET_TOKENS=${TARGET_TOKENS} ==="

python llava/eval/model_sqa3d.py \
    --model-path ChaimZhu/LLaVA-3D-7B \
    --question-file ${QUESTION_FILE} \
    --answers-file ${PRED_FILE}

python llava/eval/sqa3d_evaluator.py \
    --pred-file ${PRED_FILE} \
    --gt-file ${GT_FILE}
