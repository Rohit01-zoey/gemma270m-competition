# Merged Repository: Training + Evaluation

This repository now contains both:
1. **Training code** (original gemma270m-competition)
2. **Evaluation code** (from cs590-slm-main TA starter code)

## Environment Setup

**Prerequisites:** Python >= 3.12

### Option 1: Automated setup.py (Recommended for conda users)

```bash
# For CPU-only (development/testing)
python setup.py

# For CUDA/GPU support (training)
python setup.py --cuda

# Use mamba for faster installation (if available)
python setup.py --cuda --mamba
```

### Option 2: Using pip with pyproject.toml

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA first (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the package and all dependencies
pip install -e .

# Install cs590_eval packages
cd cs590_eval
pip install -e .
pip install -e ./ifeval
cd ..
```

### Option 3: Using requirements.txt [I used this]

```bash
# Create virtual environment
conda create -n YOURNAME python=3.12

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# Install cs590_eval packages
cd cs590_eval
pip install -e .
pip install -e ./ifeval
cd ..
```

## Directory Structure

```
gemma270m-competition/
├── cs590_eval/              # TA's evaluation code (copied from cs590-slm-main)
│   ├── eval.py              # Main evaluation script with proper timing
│   ├── generate.py          # Inference-only script
│   ├── score.py             # Scoring for all three tasks
│   ├── submit.py            # Multi-seed submission builder
│   ├── pipelines.py         # Pipeline implementations
│   ├── ifeval/              # IFEval evaluation package
│   └── data/                # Test datasets (TriviaQA, ARC-C, IFEval)
│
├── src/                     # Original training code
│   ├── sft/                 # Supervised fine-tuning (LoRA)
│   │   ├── train/           # Training scripts
│   │   ├── data/            # Data building
│   │   └── eval/            # Training evaluation
│   └── eval/                # Custom evaluation utilities
│
├── scripts/                 # Convenience scripts
│   ├── eval_model.sh        # 🔥 NEW: Evaluate trained model on all tasks
│   ├── submit_model.sh      # 🔥 NEW: Create Gradescope submission
│   ├── sft_build_data.sh    # Build training data
│   └── sft_train_lora.sh    # Train LoRA model
│
└── checkpoints/             # Your trained model checkpoints
```

## Quick Start Guide

### 1. Training a Model

Build training data:
```bash
bash scripts/sft_build_data.sh
```

Train with LoRA:
```bash
bash scripts/sft_train_lora.sh
```

### 2. Evaluating Your Trained Model

First, download the evaluation datasets (only needs to be done once):
```bash
cd cs590_eval
python download_eval_data.py
cd ..
```

This creates the test datasets in `cs590_eval/data/`:
- `triviaqa_test.jsonl`
- `arc_c_test.jsonl`
- `ifeval_test.jsonl`

Then, edit `scripts/eval_model.sh` and change the `MODEL_PATH` variable:
```bash
MODEL_PATH="checkpoints/sft/sft_final"  # Your trained model
```

Run evaluation:
```bash
bash scripts/eval_model.sh
```

This will evaluate on all three tasks (ARC-C, TriviaQA, IFEval) with proper timing.

### 3. Creating Gradescope Submission

Edit `scripts/submit_model.sh` and change the `MODEL_PATH` variable:
```bash
MODEL_PATH="checkpoints/sft/sft_final"
```

Then run:
```bash
bash scripts/submit_model.sh
```

This creates `cs590_eval/submissions.zip` with predictions from 5 different seeds.

## Key Files to Modify

### For Training
- `src/sft/train/train_sft_lora.py` - Training configuration
- `scripts/sft_train_lora.sh` - Training hyperparameters

### For Evaluation
- `cs590_eval/pipelines.py` - Modify `OurPipeline` class for custom inference
- `scripts/eval_model.sh` - Change model path and batch sizes
- `scripts/submit_model.sh` - Configure submission settings

## Important Notes

1. **Use cs590_eval/ for grading**: The evaluation code in `cs590_eval/` is the official TA code with correct timing measurements for inference. Always use this for final evaluation and submissions.

2. **Training vs Evaluation**:
   - Training: Use `src/sft/` code
   - Evaluation: Use `cs590_eval/` code

3. **Model paths**: Your trained models are in `checkpoints/`. Use the full path like `checkpoints/sft/sft_final` or `checkpoints/sft/checkpoint-2800`.

4. **Dependencies**: Run `python3 setup.py --cuda` to set up the environment with all necessary dependencies.

## Workflow Example

```bash
# 1. Train your model
bash scripts/sft_build_data.sh
bash scripts/sft_train_lora.sh

# 2. Evaluate different checkpoints
# Edit scripts/eval_model.sh to point to checkpoint-2400
bash scripts/eval_model.sh

# Edit scripts/eval_model.sh to point to checkpoint-2800
bash scripts/eval_model.sh

# 3. Create submission with best checkpoint
# Edit scripts/submit_model.sh to point to best checkpoint
bash scripts/submit_model.sh

# 4. Upload cs590_eval/submissions.zip to Gradescope