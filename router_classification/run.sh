#!/bin/bash
# Router classification training pipeline
# Usage: bash router_classification/run.sh
# To change epochs: modify --epochs value below (default: 5)

set -e

echo "=== Step 1: Prepare Data ==="
python router_classification/prepare_data.py

echo ""
echo "=== Step 2: Train Router (Classification) ==="
if [ -d "checkpoints/router_cls/final" ] && [ -f "checkpoints/router_cls/final/model.pt" ]; then
    echo "✓ Checkpoint already exists at checkpoints/router_cls/final"
    echo "  Skipping training. Delete the checkpoint to retrain."
else
    echo "Training router (classification-based)..."
    python router_classification/train.py \
        --base_model google/gemma-3-270m \
        --out_dir checkpoints/router_cls \
        --epochs 5 \
        --batch_size 16 \
        --lr 2e-5
fi

echo ""
echo "=== Step 3: Test Router ==="
python router_classification/test_router.py \
    --router_path checkpoints/router_cls/final

echo ""
echo "Done! Router saved to checkpoints/router_cls/final"
echo "Training history saved to checkpoints/router_cls/final/training_history.json"
