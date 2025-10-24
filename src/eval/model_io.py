import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


###############chk below##################

def load_model(ckpt: str, device: str | None = None):
    """
    Load tokenizer+model. ckpt can be 'google/gemma-3-270m' or a local path.
    """
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tok, model, device

@torch.no_grad()
def generate_greedy(tok, model, device, prompts: list[str], max_new_tokens=32):
    """
    Simple batched greedy decoding. Returns list[str] of generations (no prompt).
    """
    outs = []
    bs = 8
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i+bs]
        enc = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        # strip the prompt part
        for j in range(len(chunk)):
            out_ids = gen[j, enc["input_ids"].shape[1]:]
            text = tok.decode(out_ids, skip_special_tokens=True).strip()
            outs.append(text)
    return outs
