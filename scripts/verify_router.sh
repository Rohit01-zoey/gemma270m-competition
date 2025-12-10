#!/bin/bash

cd cs590_eval

echo "Normal Tasks:"
python verify_router.py
mv router_verification_results.json router_verification_normal.json 2>/dev/null || true
mv router_verification_results.csv router_verification_normal.csv 2>/dev/null || true
mv router_confusion_matrix.csv router_confusion_normal.csv 2>/dev/null || true

echo ""
echo "Hidden Tasks:"
python verify_router.py --check_hidden
mv router_verification_results.json router_verification_hidden.json 2>/dev/null || true
mv router_verification_results.csv router_verification_hidden.csv 2>/dev/null || true
mv router_confusion_matrix.csv router_confusion_hidden.csv 2>/dev/null || true
