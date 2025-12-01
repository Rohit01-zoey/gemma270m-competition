#!/bin/bash
#SBATCH --job-name=stage1_IT
#SBATCH --time=10:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:a6000:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --mail-user=rkv6@duke.edu
#SBATCH --output=stage1_v2.out
#SBATCH --mail-type=END,FAIL

torchrun --nproc_per_node=1 stage1_v2.py

# python stage1.py \
#   --model_name google/gemma-3-270m \
#   --train_path data/stage1/stage1_train.jsonl \
#   --eval_path data/stage1/stage1_eval.jsonl \
#   --output_dir ckpts/stage1_sft \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 8 \
#   --num_train_epochs 1.5 \
#   --bf16


# torchrun --nproc_per_node=2 train_stage1.py \
#   --model_name google/gemma-3-270m \
#   --train_path data/stage1_train.jsonl \
#   --eval_path data/stage1_eval.jsonl \
#   --output_dir ckpts/stage1_sft \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 4 \
#   --num_train_epochs 1.5 \
#   --bf16
