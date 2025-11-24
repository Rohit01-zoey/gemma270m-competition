#!/bin/bash
# Script to evaluate a trained model on all three tasks
#
# Usage:
#   1. Edit the MODEL_PATH below to point to your trained model
#   2. Run: bash scripts/eval_model.sh

set -e  # Exit on error

#####################################
# CONFIGURE YOUR MODEL PATH HERE
#####################################
# MODEL_PATH="checkpoints/sft/sft_final"
# Or use a checkpoint:
# MODEL_PATH="checkpoints/sft/checkpoint-2800"
# Or use the base model:
MODEL_PATH="google/gemma-3-270m"

#####################################
# EVALUATION SETTINGS
#####################################
BATCH_SIZE=512
MAX_NEW_TOKENS=512
DATA_SIZE=-1 # -1 for full dataset

#####################################
# Run evaluations
#####################################
echo "========================================"
echo "Evaluating model: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Data size: $DATA_SIZE"
echo "========================================"
echo ""

cd cs590_eval

echo ">>> Running ARC-C evaluation..."
python eval.py \
  --task arc-c \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo ">>> Running TriviaQA evaluation..."
python eval.py \
  --task triviaqa \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo ">>> Running IFEval evaluation..."
python eval.py \
  --task ifeval \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo "========================================"
echo "Evaluation complete! Results saved in cs590_eval/outputs/"
echo "========================================"
