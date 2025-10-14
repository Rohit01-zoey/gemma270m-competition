#!/usr/bin/env python3
"""
setup.py — bootstrap the conda environment for this repo.

Usage:
  python setup.py                 # create/update env 'gemma270m', CPU PyTorch
  python setup.py --env gemma270m-dev
  python setup.py --cuda          # install CUDA PyTorch build (Linux w/ NVIDIA)
  python setup.py --mamba         # use mamba if available
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def run(cmd, **kw):
    print(f"➜  {' '.join(cmd)}")
    subprocess.check_call(cmd, **kw)

def have(cmd):
    return shutil.which(cmd) is not None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="gemma270m", help="Conda env name")
    ap.add_argument("--cuda", action="store_true",
                    help="Install CUDA PyTorch build (Linux + NVIDIA). Default is CPU build.")
    ap.add_argument("--mamba", action="store_true", help="Prefer mamba if available")
    args = ap.parse_args()

    # 0) Sanity checks
    conda_exe = shutil.which("conda")
    if not conda_exe:
        sys.exit("ERROR: conda not found on PATH. Install Miniconda/Anaconda first.")

    solver = "conda"
    if args.mamba and have("mamba"):
        solver = "mamba"

    # 1) Create/update env from environment.yml
    env_yml = os.path.join(HERE, "environment.yml")
    if not os.path.exists(env_yml):
        sys.exit("ERROR: environment.yml not found at repo root.")
    try:
        # If env exists, update; else create
        out = subprocess.run(
            [conda_exe, "env", "list"], capture_output=True, text=True, check=True
        ).stdout
        exists = any(line.split()[0] == args.env for line in out.splitlines() if line.strip())
        if exists:
            run([solver, "env", "update", "-n", args.env, "-f", env_yml, "--prune"])
        else:
            run([solver, "env", "create", "-n", args.env, "-f", env_yml])
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

    # 2) Install PyTorch appropriate build INSIDE the env
    #    We use pip because it's the most predictable across OS.
    #    CPU build by default; CUDA if requested and platform looks right.
    pip_cmd = [conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"]
    run(pip_cmd)

    torch_install = [conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install"]
    if args.cuda and platform.system() == "Linux":
        # CUDA build from PyTorch index; picks latest stable (e.g., cu121)
        torch_line = ["--index-url", "https://download.pytorch.org/whl/cu121", "torch", "torchvision", "torchaudio"]
    else:
        # CPU build (works on Linux/macOS/Windows)
        torch_line = ["torch", "torchvision", "torchaudio"]
    run(torch_install + torch_line)

    # 3) Install this repo in editable mode
    run([conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install", "-e", HERE])

    # 4) Print activation hint (cannot activate caller shell from a child process)
    print("\n✅ Environment ready.")
    print(f"👉 Activate it with:\n\n    conda activate {args.env}\n")
    print("Quick checks:")
    print(f"    python -c \"import torch, transformers; print(torch.__version__); print(transformers.__version__)\"")

if __name__ == "__main__":
    main()
