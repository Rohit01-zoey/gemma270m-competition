#!/bin/bash
#
# Evaluate all checkpoints in a directory to find the best one
# Uses cs590_eval with self-consistency for improved accuracy
#
# Usage:
#   bash scripts/eval_sweep_checkpoints_sc.sh <checkpoint_directory> [num_samples]
#
# Examples:
#   bash scripts/eval_sweep_checkpoints_sc.sh checkpoints/sft_multitask
#   bash scripts/eval_sweep_checkpoints_sc.sh checkpoints/sft_multitask 5
#   bash scripts/eval_sweep_checkpoints_sc.sh checkpoints/sft_arc_only 7
#

set -e  # Exit on error

#####################################
# PARSE ARGUMENTS
#####################################
if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/eval_sweep_checkpoints_sc.sh <checkpoint_directory> [num_samples]"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_directory  - Directory containing checkpoints to evaluate"
    echo "  num_samples          - Number of samples for self-consistency (default: 5)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval_sweep_checkpoints_sc.sh checkpoints/sft_multitask"
    echo "  bash scripts/eval_sweep_checkpoints_sc.sh checkpoints/sft_multitask 7"
    exit 1
fi

CKPT_DIR=$1
NUM_SAMPLES=${2:-5}  # Default to 5 samples if not specified

# Validate directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Directory '$CKPT_DIR' does not exist"
    exit 1
fi

# Validate num_samples is a positive integer
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]] || [ "$NUM_SAMPLES" -lt 1 ]; then
    echo "Error: num_samples must be a positive integer (got: '$NUM_SAMPLES')"
    exit 1
fi

#####################################
# CONFIGURATION
#####################################
# Dev/validation data (NOT test data - that's for final evaluation only)
DEV_DATA="data/sft_dev.jsonl"
BATCH_SIZE=64
MAX_NEW_TOKENS=512

# Output CSV filename includes num_samples
if [ "$NUM_SAMPLES" -eq 1 ]; then
    OUTPUT_CSV="${CKPT_DIR}/sweep_results_cs590.csv"
else
    OUTPUT_CSV="${CKPT_DIR}/sweep_results_cs590_sc${NUM_SAMPLES}.csv"
fi

#####################################
# RUN CHECKPOINT SWEEP
#####################################
echo "=========================================================="
echo "Checkpoint Sweep with Self-Consistency"
echo "=========================================================="
echo "Checkpoint Directory: $CKPT_DIR"
echo "Dev/Val Data:         $DEV_DATA"
echo "Num Samples:          $NUM_SAMPLES"
echo "Batch Size:           $BATCH_SIZE"
echo "Max New Tokens:       $MAX_NEW_TOKENS"
echo "Output CSV:           $OUTPUT_CSV"
echo "=========================================================="
echo ""

if [ "$NUM_SAMPLES" -eq 1 ]; then
    echo "⚠️  Self-consistency is DISABLED (num-samples=1)"
    echo "   This is equivalent to standard evaluation"
else
    echo "✓ Self-consistency ENABLED with $NUM_SAMPLES samples per input"
    echo "   This will take approximately ${NUM_SAMPLES}x longer per checkpoint"
fi

echo ""
echo "⚠️  Evaluating on VALIDATION data (not test data)"
echo "   Using cs590_eval for Gradescope-compatible scoring"
echo ""

python -m src.sft.eval.sweep_checkpoints_cs590_sc \
    --ckpt_root "$CKPT_DIR" \
    --test_data "$DEV_DATA" \
    --batch_size $BATCH_SIZE \
    --max_new_tokens $MAX_NEW_TOKENS \
    --num-samples $NUM_SAMPLES \
    --out_csv "$OUTPUT_CSV"

#####################################
# DISPLAY RESULTS
#####################################
echo ""
echo "=========================================================="
echo "Evaluation Complete!"
echo "=========================================================="
echo "Results saved to: $OUTPUT_CSV"
echo ""
echo "To view the best checkpoint:"
echo "  cat ${CKPT_DIR}/best_checkpoint.txt"
echo ""
echo "To sort by ARC accuracy:"
echo "  sort -t',' -k2 -rn $OUTPUT_CSV | head -5"
echo ""
echo "Top 3 checkpoints by ARC accuracy:"
echo "----------------------------------------------------------"
sort -t',' -k2 -rn "$OUTPUT_CSV" | head -4 | tail -3
echo "=========================================================="
