#!/usr/bin/env python3
import os, sys, shutil, subprocess, argparse, platform

# ---- Detect setuptools invocation (pip calling egg_info, build, sdist, etc.) ----
SETUPTOOLS_COMMANDS = {"egg_info","build","bdist_wheel","sdist","install","develop"}

if any(cmd in sys.argv for cmd in SETUPTOOLS_COMMANDS):
    # Minimal packaging fallback so pip -e . works
    from setuptools import setup, find_packages
    setup(
        name="gemma270m-competitions",
        version="0.1.0",
        description="Training & eval pipeline for Gemma-3-270M on TriviaQA, ARC-C, IFEval",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
    )
    sys.exit(0)

# ---- Otherwise: run the bootstrap logic (previous script) ----
def run(cmd, **kw):
    print(f"➜  {' '.join(cmd)}")
    subprocess.check_call(cmd, **kw)

def have(cmd):
    return shutil.which(cmd) is not None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="gemma270m", help="Conda env name")
    ap.add_argument("--cuda", action="store_true", help="Install CUDA PyTorch build")
    ap.add_argument("--mamba", action="store_true", help="Prefer mamba if available")
    args = ap.parse_args()

    conda_exe = shutil.which("conda")
    if not conda_exe:
        sys.exit("ERROR: conda not found on PATH.")

    solver = "conda"
    if args.mamba and have("mamba"):
        solver = "mamba"

    here = os.path.dirname(os.path.abspath(__file__))
    env_yml = os.path.join(here, "environment.yml")
    if not os.path.exists(env_yml):
        sys.exit("ERROR: environment.yml not found.")

    out = subprocess.run([conda_exe, "env", "list"], capture_output=True, text=True, check=True).stdout
    exists = any(line.split()[0] == args.env for line in out.splitlines() if line.strip())
    if exists:
        run([solver, "env", "update", "-n", args.env, "-f", env_yml, "--prune"])
    else:
        run([solver, "env", "create", "-n", args.env, "-f", env_yml])

    run([conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    torch_cmd = [conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install"]
    if args.cuda and platform.system() == "Linux":
        torch_cmd += ["--index-url", "https://download.pytorch.org/whl/cu121", "torch", "torchvision", "torchaudio"]
    else:
        torch_cmd += ["torch", "torchvision", "torchaudio"]
    run(torch_cmd)

    # run([conda_exe, "run", "-n", args.env, "python", "-m", "pip", "install", "-e", here])

    print("\n✅ Environment ready.")
    print(f"👉 Activate with: conda activate {args.env}")

if __name__ == "__main__":
    main()
