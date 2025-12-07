python -m  src.sft.data.build_sft_dataset_v2 \
  --arc 12000 \
  --trivia 10000 \
  --flan 12000 \
  --openbookqa 5000 --qasc 5000 \
  --squad 8000 --nqopen 6000 --boolq 5000 \
  --hellaswag 5000 \
  --dev_ratio 0.1 \
  --out-train data/sft_train_v2.jsonl \
  --out-dev data/sft_dev_v2.jsonl
