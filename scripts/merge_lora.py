# scripts/merge_lora.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, sys

base_id = "google/gemma-3-270m"
adapter_dir = "checkpoints/sft_continued/checkpoint-1600"
out_dir = "checkpoints/sft_continued/merged-1600"

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map={"":0})
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()         
os.makedirs(out_dir, exist_ok=True)
merged.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
print("Merged to:", out_dir)
