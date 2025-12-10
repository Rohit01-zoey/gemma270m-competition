#!/bin/bash
# Generate submissions.zip for Gradescope
set -e

cd cs590_eval

python submit.py \
  --pipeline our \
  --seeds 1,2,3,4,5 \
  --data_size 1000 \
  --out_dir submissions_cached

echo "Done! Upload submissions.zip to Gradescope"
