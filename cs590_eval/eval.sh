#!/bin/bash

python eval.py \
  --task arc-c \
  --model google/gemma-3-270m-it \
  --data-size -1 \
  --batch-size 512 \
  --max-new-tokens 512 \
  --skip-existing

python eval.py \
  --task triviaqa \
  --model google/gemma-3-270m-it \
  --data-size -1 \
  --batch-size 512 \
  --max-new-tokens 512 \
  --skip-existing

python eval.py \
  --task ifeval \
  --model google/gemma-3-270m-it \
  --data-size -1 \
  --batch-size 512 \
  --max-new-tokens 512 \
  --skip-existing