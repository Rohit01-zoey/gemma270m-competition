python -m src.sft.train.train_sft_lora ^
  --model_name google/gemma-3-270m ^
  --train_file data/sft_train.jsonl ^
  --out_dir checkpoints/sft ^
  --num_train_epochs 2 ^
  --per_device_train_batch_size 8 ^
  --grad_accum_steps 2 ^
  --max_len 256 ^
  --bf16 ^
  --save_steps 200
