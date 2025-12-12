#!/bin/bash
# Script to evaluate using MultiModelRouterPipeline with task-specific models
#
# Usage:
#   1. Edit cs590_eval/model_config.py to configure your models per task
#   2. Run: bash scripts/eval_multimodel.sh

set -e  # Exit on error

#####################################
# EVALUATION SETTINGS
#####################################
BATCH_SIZE=512
MAX_NEW_TOKENS=512
DATA_SIZE=1000 # -1 for full dataset

#####################################
# Run evaluations
#####################################
echo "========================================"
echo "Evaluating with MultiModelRouterPipeline"
echo "Models configured in cs590_eval/model_config.py"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Data size: $DATA_SIZE"
echo "========================================"
echo ""

cd cs590_eval

echo ">>> Running ARC-C evaluation..."
python eval_multimodel.py \
  --task arc-c \
  --out_dir outputs_multimodel_1212_01/ \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo ">>> Running TriviaQA evaluation..."
python eval_multimodel.py \
  --task triviaqa \
  --out_dir outputs_multimodel_1212_01/ \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo ">>> Running IFEval evaluation..."
python eval_multimodel.py \
  --task ifeval \
  --out_dir outputs_multimodel_1212_01/ \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --skip-existing

echo ""
echo "========================================"
echo "Evaluation complete! Results saved in cs590_eval/outputs_multimodel/"
echo "========================================"
