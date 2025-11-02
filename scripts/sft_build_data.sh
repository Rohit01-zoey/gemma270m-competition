python -m src.sft.data.build_sft_dataset \
  --out-train data/sft_train.jsonl \
  --out-dev   data/sft_dev.jsonl \
  --arc 8000 --trivia 8000 --flan 16000 \
  --dev_ratio 0.1