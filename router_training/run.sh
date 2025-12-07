#!/bin/bash
# Router training pipeline (generation-based)

set -e

echo "=== Step 1: Prepare Data ==="
python router_training/prepare_data.py

echo ""
echo "=== Step 2: Train Router ==="
if [ -d "checkpoints/router/final" ] && [ -f "checkpoints/router/final/pytorch_model.bin" ]; then
    echo "✓ Checkpoint already exists at checkpoints/router/final"
    echo "  Skipping training. Delete the checkpoint to retrain."
else
    echo "Training router (generation-based)..."
    python router_training/train.py \
        --base_model google/gemma-3-270m \
        --out_dir checkpoints/router \
        --epochs 3 \
        --batch_size 16 \
        --lr 2e-5 \
        --use_lora
fi

echo ""
echo "=== Step 3: Test Router ==="
python router_training/test_router.py

echo ""
echo "Done! Router saved to checkpoints/router/final"
