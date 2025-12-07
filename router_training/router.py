"""
Router inference.

Usage:
    from router_training.router import Router

    router = Router("checkpoints/router/final")
    task = router.classify("What is 2+2?")
    print(task)  # "factual_qa"
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Router:
    TASKS = {"reasoning", "factual_qa", "instruction_following"}
    DEFAULT = "factual_qa"

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(self, question):
        return f"Classify task type:\n{question.strip()}\nTask:"

    @torch.inference_mode()
    def classify(self, question):
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )

        text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        return self._parse_output(text)

    @torch.inference_mode()
    def classify_batch(self, questions, batch_size=16):
        results = []
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i+batch_size]
            prompts = [self.format_prompt(q) for q in batch]

            inputs = self.tokenizer(
                prompts, return_tensors="pt",
                padding=True, truncation=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

            for j, output in enumerate(outputs):
                input_len = inputs.input_ids[j].shape[0]
                text = self.tokenizer.decode(
                    output[input_len:],
                    skip_special_tokens=True
                ).strip().lower()
                results.append(self._parse_output(text))

        return results

    def _parse_output(self, text):
        # Direct match
        if text in self.TASKS:
            return text

        # Contains match
        for task in self.TASKS:
            if task in text:
                return task

        # Partial matches
        if "reason" in text or "arc" in text:
            return "reasoning"
        if "factual" in text or "trivia" in text or "qa" in text:
            return "factual_qa"
        if "instruction" in text or "ifeval" in text or "follow" in text:
            return "instruction_following"

        print(f"Warning: Could not parse '{text}', using default '{self.DEFAULT}'")
        return self.DEFAULT
