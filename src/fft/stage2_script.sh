#!/bin/bash
#SBATCH --job-name=stage2_SFT
#SBATCH --time=06:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=rkv6@duke.edu
#SBATCH --output=stage2_v2_arc_only.out
#SBATCH --mail-type=END,FAIL

torchrun --nproc_per_node=4 stage2_v2.py

# python stage2.py \
#   --model_name ckpts/stage1_sft \
#   --train_path data/stage2/stage2_train.jsonl \
#   --eval_path data/stage2/stage2_eval.jsonl \
#   --output_dir ckpts/stage2_sft \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --num_train_epochs 1.0 \
#   --bf16


# torchrun --nproc_per_node=2 stage2.py \
#   --model_name ckpts/stage1_sft \
#   --train_path data/stage2/stage2_train.jsonl \
#   --eval_path data/stage2/stage2_eval.jsonl \
#   --output_dir ckpts/stage2_sft \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --num_train_epochs 1.0 \
#   --bf16
