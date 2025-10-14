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


