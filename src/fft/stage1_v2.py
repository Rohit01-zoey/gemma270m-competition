from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    # dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
    dataset = load_dataset("HuggingFaceTB/smol-smoltalk")

    # Configure model and tokenizer
    model_name = "google/gemma-3-270m"
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name).to(
        device
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    # Setup chat template
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Configure trainer
    training_args = SFTConfig(
        output_dir="./sft_output",
        max_steps=20000,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        logging_steps=10,
        save_steps=5000,
        eval_strategy="steps",
        eval_steps=1000,
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()