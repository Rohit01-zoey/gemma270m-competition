# File for evaluating Gemma model on the ARC-C challenge
# run using eval/script/arc_c.sh 
import argparse
from datasets import load_dataset
from model_io import load_model, generate_greedy
from prompt_templates import ARC
from postprocess import extract_letter
from preprocess import choices_map

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="google/gemma-3-270m")
    ap.add_argument("--split", default="test", choices=["validation","test","train"])
    ap.add_argument("--limit", type=int, default=None) 
    ap.add_argument("--max_new_tokens", type=int, default=4)
    ################ decoding parameters #################
    ############### logging parameters ##################
    args = ap.parse_args()

    tok, model, device = load_model(args.ckpt)
    ds = load_dataset("ai2_arc", "ARC-Challenge")[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = []
    answers = []
    for ex in ds:
        cm = choices_map(ex)
        prompt = ARC.format(
            q=ex["question"],
            A=cm.get("A", ""),
            B=cm.get("B", ""),
            C=cm.get("C", ""),
            D=cm.get("D", "")
        )
        prompts.append(prompt)
        answers.append(ex["answerKey"].strip().upper())

    gens = generate_greedy(tok, model, device, prompts, max_new_tokens=args.max_new_tokens)
    preds = [extract_letter(g) or "" for g in gens]
    acc = sum(int(p == g) for p, g in zip(preds, answers)) / len(answers)

    print(f"ARC-Challenge ({args.split}) with N={len(preds)} samples")
    print(f"Accuracy: {acc:.4f}")

    ######## logging code below ##########

if __name__ == "__main__":
    main()
