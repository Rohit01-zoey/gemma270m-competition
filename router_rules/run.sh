#!/bin/bash
# Rule-based router evaluation pipeline
# No training needed - purely keyword/heuristic based

set -e

echo "=== Step 1: Prepare Data ==="
python router_rules/prepare_data.py

echo ""
echo "=== Step 2: Test Router (Rule-based) ==="
echo "ℹ️  Rule-based router requires no training"
python router_rules/test_router.py

echo ""
echo "Done! Rule-based router evaluation complete"
echo "Results saved to router_rules/results/"
