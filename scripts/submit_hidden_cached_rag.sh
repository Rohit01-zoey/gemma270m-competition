#!/bin/bash
# Generate submissions.zip for Gradescope
set -e

cd cs590_eval

python submit.py \
  --pipeline our \
  --seeds 1,2,3 \
  --data_size 1000 \
  --submit_hidden \
  --out_dir submissions_cached

echo "Done! Upload submissions_cached_hidden.zip to Gradescope"
