#!/bin/bash
# Script to evaluate a trained model on all three tasks with self-consistency
#
# Usage:
#   1. Edit the MODEL_PATH below to point to your trained model
#   2. Edit NUM_SAMPLES for self-consistency (1=disabled, 5+ recommended)
#   3. Run: bash scripts/eval_model_sc.sh
#
# Example:
#   bash scripts/eval_model_sc.sh

set -e  # Exit on error

#####################################
# CONFIGURE YOUR MODEL PATH HERE
#####################################
# MODEL_PATH="checkpoints/sft/sft_final"
# Or use a checkpoint:
# MODEL_PATH="checkpoints/sft/checkpoint-2800"
# Or use the base model:
MODEL_PATH="google/gemma-3-270m-it"

#####################################
# SELF-CONSISTENCY SETTINGS
#####################################
# Number of samples to generate per input for self-consistency
# 1 = disabled (standard evaluation)
# 5-7 = recommended for good balance
# 10+ = better accuracy but much slower
NUM_SAMPLES=20

#####################################
# EVALUATION SETTINGS
#####################################
BATCH_SIZE=512
MAX_NEW_TOKENS=512
DATA_SIZE=-1 # -1 for full dataset
TEMPERATURE=0.7
TOP_P=0.95

#####################################
# Display configuration
#####################################
echo "========================================"
echo "Evaluating model with Self-Consistency"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Num samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Data size: $DATA_SIZE"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "========================================"
echo ""

if [ "$NUM_SAMPLES" -eq 1 ]; then
    echo "⚠️  Self-consistency is DISABLED (num-samples=1)"
    echo "   Set NUM_SAMPLES to 5+ to enable self-consistency"
else
    echo "✓ Self-consistency ENABLED with $NUM_SAMPLES samples per input"
    echo "   NOTE: Self-consistency is ONLY implemented for ARC-C"
    echo "   TriviaQA and IFEval will use standard single-sample evaluation"
    echo "   ARC-C evaluation will take approximately ${NUM_SAMPLES}x longer"
fi
echo ""

#####################################
# Run evaluations
#####################################
cd cs590_eval

echo ">>> Running ARC-C evaluation..."
python eval_sc.py \
  --task arc-c \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --skip-existing

echo ""
echo ">>> Running TriviaQA evaluation..."
python eval_sc.py \
  --task triviaqa \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --skip-existing

echo ""
echo ">>> Running IFEval evaluation..."
python eval_sc.py \
  --task ifeval \
  --model "$MODEL_PATH" \
  --data-size "$DATA_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --num-samples "$NUM_SAMPLES" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --skip-existing

cd ..

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved in: cs590_eval/outputs/"
echo ""
echo "To view results:"
if [ "$NUM_SAMPLES" -eq 1 ]; then
    echo "  ls -lh cs590_eval/outputs/*_metrics.json"
else
    echo "  ls -lh cs590_eval/outputs/*_sc${NUM_SAMPLES}_metrics.json"
fi
echo ""
echo "To view metrics:"
if [ "$NUM_SAMPLES" -eq 1 ]; then
    echo "  cat cs590_eval/outputs/*arc_c*_metrics.json"
    echo "  cat cs590_eval/outputs/*triviaqa*_metrics.json"
    echo "  cat cs590_eval/outputs/*ifeval*_metrics.json"
else
    echo "  cat cs590_eval/outputs/*arc_c*_sc${NUM_SAMPLES}_metrics.json"
    echo "  cat cs590_eval/outputs/*triviaqa*_sc${NUM_SAMPLES}_metrics.json"
    echo "  cat cs590_eval/outputs/*ifeval*_sc${NUM_SAMPLES}_metrics.json"
fi
echo "========================================"
