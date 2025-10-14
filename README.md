# gemma270m-competition


## 🚀 Setting Up the Environment

Follow these steps to set up the Gemma-3-270M benchmarking environment:

1. **Clone the repository**
   ```bash
   git clone https://github.com/rohit01-zoey/gemma270m-competition.git
   cd gemma270m-competition
   ```

2. **Create the Conda environment**

   Run the setup script to automatically create a Conda environment (`gemma270m`) and install all dependencies.

   ```bash
   python3 setup.py --cuda
   ```

   💡 **Tip:** Use the `--cuda` flag if your system has an NVIDIA GPU.  
   This installs CUDA-compatible PyTorch builds (`torch`, `torchvision`, `torchaudio`).

   If you are on CPU-only or macOS, simply run:

   ```bash
   python3 setup.py
   ```

3. **Activate the environment**
   ```bash
   conda activate gemma270m
   ```

4. **Verify installation**
   ```bash
   python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
   ```

   You should see valid version numbers printed without any errors.

---

### ✅ Running Baseline Benchmarks

Once activated, you can run the benchmark scripts directly to evaluate the **vanilla Gemma-3-270M** model on the three datasets:

```bash
python -m src.eval.eval_triviaqa --ckpt google/gemma-3-270m
python -m src.eval.eval_arc_c --ckpt google/gemma-3-270m
python -m src.eval.eval_ifeval --ckpt google/gemma-3-270m
```

Each script loads the dataset automatically and reports the model’s baseline performance:

- **TriviaQA** → Exact Match (EM) / F1  
- **ARC-Challenge** → Multiple-choice Accuracy  
- **IFEval** → Instruction-following rule compliance rate  

---

### 🧩 Notes

- The setup script installs everything needed for inference and benchmarking, including Hugging Face Transformers, Datasets, Accelerate, and PyTorch.  
- You can re-run `python3 setup.py --cuda` anytime to update dependencies or switch to a GPU build.  
- For custom models or fine-tuned checkpoints, replace `--ckpt google/gemma-3-270m` with your local path.

