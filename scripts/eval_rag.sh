#!/bin/bash
#SBATCH --job-name=rag_eval
##SBATCH --output=/home/users/ms1254/gemma270m-competition/logs/rag_full_msmarco_%j.log 
##SBATCH --error=/home/users/ms1254/gemma270m-competition/logs/rag_full_msmarco_%j.err 
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH --time=10:00:00

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gemma270m

# Go to eval directory
cd ~/gemma270m-competition/cs590_eval


echo "========================================"
echo "RAG Evaluation Started"
echo "========================================"

# Run TriviaQA with RAG (small test first)
echo ">>> Running TriviaQA with RAG..."
python eval_rag.py \
  --task triviaqa \
  --model google/gemma-3-270m \
  --data-size 100 \
  --top-k 3 \
  --use-full-msmarco \
  --output-prefix "triviaqa_rag_msmarco_n100_k3_base_nostem"


#echo ""
#echo ">>> Running ARC-C with RAG..."
#python eval_rag.py \
#  --task arc-c \
#  --model google/gemma-3-270m \
#  --data-size 100 \
#  --top-k 3 \
#  --use-full-msmarco \
#  --output-prefix "arc_rag_msmarco_n100_k3_base_nostem"

echo ""
echo "========================================"
echo "RAG Evaluation Complete!"
echo "Results saved in cs590_eval/outputs/"
echo "========================================"
