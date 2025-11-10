#!/bin/bash
# Script to create submissions.zip for Gradescope
#
# Usage:
#   1. Edit the MODEL_PATH below to point to your trained model
#   2. Run: bash scripts/submit_model.sh
#   3. Upload the generated submissions.zip to Gradescope

set -e  # Exit on error

#####################################
# CONFIGURE YOUR MODEL PATH HERE
#####################################
MODEL_PATH="checkpoints/sft/sft_final"
# Or use a checkpoint:
# MODEL_PATH="checkpoints/sft/checkpoint-2800"

#####################################
# SUBMISSION SETTINGS
#####################################
PIPELINE="our"  # Use "our" pipeline from pipelines.py
SEEDS="1,2,3,4,5"  # Run 5 different seeds for reproducibility
DATA_SIZE=1000

#####################################
# Create submission
#####################################
echo "========================================"
echo "Creating submission for: $MODEL_PATH"
echo "Pipeline: $PIPELINE"
echo "Seeds: $SEEDS"
echo "========================================"
echo ""

cd cs590_eval

python submit.py \
  --model "$MODEL_PATH" \
  --pipeline "$PIPELINE" \
  --seeds "$SEEDS" \
  --data_size "$DATA_SIZE"

echo ""
echo "========================================"
echo "✅ Submission created: cs590_eval/submissions.zip"
echo "📤 Upload this file to Gradescope!"
echo "========================================"
