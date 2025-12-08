from typing import Any, Dict, List, Optional

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm
from collections import Counter
from rule_based_router import route_question
from model_config import MODEL_MAP

# try:
#     from vllm import LLM, SamplingParams
#     VLLM_AVAILABLE = True
# except Exception:
#     VLLM_AVAILABLE = False


# ---------------- helpers ----------------
def apply_instruct_template(prompt, tokenizer: PreTrainedTokenizerBase) -> str:
    has_chat = tokenizer and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template
    if has_chat:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, list):
        lines: List[str] = []
        for m in prompt:
            role = m.get("role", "user")
            prefix = "User" if role == "user" else ("Assistant" if role == "assistant" else role.capitalize())
            lines.append(f"{prefix}: {m.get('content', '')}")
        return "\n".join(lines)
    return str(prompt)


def _strip_code_fences(text: str) -> str:
    text = re.sub(r'^\s*```.*$', '', text, flags=re.MULTILINE)
    return text.strip()


@torch.no_grad()
def _batched_generate_hf(
    model: Any,
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
    *,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 16,
) -> List[str]:
    outs: List[str] = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        dec = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        outs.extend([d.strip() for d in dec])
    return outs


# def _batched_generate_vllm(
#     model: Any,
#     tokenizer: PreTrainedTokenizerBase,
#     prompts: List[str],
#     *,
#     max_new_tokens: int = 128,
#     do_sample: bool = False,
#     temperature: float = 0.7,
#     top_p: float = 0.95,
# ) -> List[str]:
#     sampling_params = SamplingParams(
#         max_tokens=max_new_tokens,
#         temperature=temperature if do_sample else 0.0,
#         top_p=top_p if do_sample else 1.0,
#         skip_special_tokens=True,
#     )
#     outputs = model.generate(prompts, sampling_params)
#     return [o.outputs[0].text.strip() for o in outputs]


# ---------------- logic-only processors ----------------
class ProcessLogic:
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        return [apply_instruct_template(str(it.get("prompt", "")), tokenizer) for it in items]

    def postprocess(self, outputs: List[str], items: List[Dict[str, Any]]) -> List[str]:
        # default: strip code fences and whitespace
        return [_strip_code_fences(t).strip() for t in outputs]


class FactualQAProcessor(ProcessLogic):
    def __init__(self, *, few_shot: int = 0):
        self.few_shot = max(0, min(few_shot, 5))

    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            q = it.get("question") or it.get("query") or ""
            if self.few_shot <= 0:
                messages = [{"role": "user", "content": f"Answer the question concisely.\nQuestion: {q}\nAnswer:"}]
                prompts.append(apply_instruct_template(messages, tokenizer))
            else:

                examples = it.get("few_shot_examples", [])[: self.few_shot]
                parts: List[str] = ["Answer the question concisely."]
                for ex in examples:
                    ex_q = ex.get("question", "")
                    ex_a = ex.get("answer", "")
                    parts.append(f"Question: {ex_q}\nAnswer: {ex_a}")
                parts.append(f"Question: {q}\nAnswer:")
                messages = [{"role": "user", "content": "\n".join(parts)}]
                prompts.append(apply_instruct_template(messages, tokenizer))
        return prompts

    def postprocess(self, outputs: List[str], items: List[Dict[str, Any]]) -> List[str]:
        cleaned = super().postprocess(outputs, items)
        return [c.split("\n")[0].strip() for c in cleaned]


class ReasoningProcessor(ProcessLogic):
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts: List[str] = []
        for it in items:
            question = it.get("question", "")
            content =  "Please reason and choose the correct answer for the following question:\n"
            content += question.strip()
            prompts.append(apply_instruct_template([{ "role": "user", "content": content }], tokenizer))
        return prompts





class InstructionFollowingProcessor(ProcessLogic):
    def preprocess(self, items: List[Dict[str, Any]], tokenizer: PreTrainedTokenizerBase) -> List[str]:
        prompts = [str(it.get("prompt", "")) for it in items]
        # apply chat template
        return [apply_instruct_template(p, tokenizer) for p in prompts]


# ---------------- base inference pipeline (loads model, uses processors) ----------------
class BasePipeline:
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.model.config, "pad_token_id", None) is None and self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _generate(self, prompts: List[str], *, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float, batch_size: int) -> List[str]:
        # Prompts are already formatted; do not apply chat template here
        return _batched_generate_hf(
            self.model,
            self.tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )

    def run(
        self,
        items: List[Dict[str, Any]],
        *,
        logic: Optional[ProcessLogic] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if logic is None:
            logic = ProcessLogic()
        prompts = logic.preprocess(items, self.tokenizer)
        raw = self._generate(
            prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )
        return logic.postprocess(raw, items)




class RouterPipeline(BasePipeline):
    def __init__(self, model_name: str):
        super().__init__(model_name)

    def _router(self, item: Dict[str, Any]) -> str:
        """
        Route questions to task type using rule-based router.
        Uses keyword matching to classify: reasoning, factual_qa, or instruction_following
        """
        return route_question(item)

    def run_with_router(
        self,
        items: List[Dict[str, Any]],
        router_fn,
        logic_map: Optional[Dict[str, ProcessLogic]] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        # Group items by route key
        groups: Dict[str, List[int]] = {}
        for idx, it in enumerate(items):
            key = router_fn(it)
            groups.setdefault(key, []).append(idx)

        # Default logic map if not provided
        if logic_map is None:
            logic_map = {
                "factual_qa": FactualQAProcessor(),
                "reasoning": ReasoningProcessor(),
                "instruction_following": InstructionFollowingProcessor(),
            }

        outputs: List[Optional[str]] = [None] * len(items)
        for key, idxs in groups.items():
            sub_items = [items[i] for i in idxs]
            logic = logic_map.get(key)
            if logic is None:
                logic = FactualQAProcessor()
            preds = super().run(
                sub_items,
                logic=logic,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            for i, p in zip(idxs, preds):
                outputs[i] = p
        return [o if o is not None else "" for o in outputs]

    def run(self, items: List[Dict[str, Any]],
            logic: Optional[ProcessLogic] = None,
            **gen_kwargs) -> List[str]:
        return self.run_with_router(items, self._router, **gen_kwargs)


class MultiModelRouterPipeline(BasePipeline):
    """
    Router pipeline that uses different models for different task types.
    Models are configured in model_config.py MODEL_MAP.
    """
    def __init__(self, model_name: str = None):
        """
        Initialize with task-specific models from MODEL_MAP.

        Args:
            model_name: Ignored (kept for compatibility), uses MODEL_MAP instead
        """
        # Don't call super().__init__ - we'll manage multiple models
        self.task_models = {}
        self.task_tokenizers = {}

        print("Loading task-specific models from MODEL_MAP...")
        for task_type, model_path in MODEL_MAP.items():
            print(f"  {task_type}: {model_path}")
            self.task_models[task_type] = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.task_tokenizers[task_type] = AutoTokenizer.from_pretrained(model_path)
            if self.task_tokenizers[task_type].pad_token is None:
                self.task_tokenizers[task_type].pad_token = self.task_tokenizers[task_type].eos_token

    def _router(self, item: Dict[str, Any]) -> str:
        """Route questions to task type using rule-based router."""
        return route_question(item)

    def _generate_for_task(self, task_type: str, prompts: List[str],
                          max_new_tokens: int, do_sample: bool,
                          temperature: float, top_p: float, batch_size: int) -> List[str]:
        """Generate using task-specific model."""
        model = self.task_models.get(task_type, self.task_models["factual_qa"])
        tokenizer = self.task_tokenizers.get(task_type, self.task_tokenizers["factual_qa"])

        return _batched_generate_hf(
            model, tokenizer, prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            batch_size=batch_size,
        )

    def run(
        self,
        items: List[Dict[str, Any]],
        logic: Optional[ProcessLogic] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        # Group items by task type
        groups: Dict[str, List[int]] = {}
        for idx, item in enumerate(items):
            task_type = self._router(item)
            groups.setdefault(task_type, []).append(idx)

        # Default logic map
        logic_map = {
            "factual_qa": FactualQAProcessor(),
            "reasoning": ReasoningProcessor(),
            "instruction_following": InstructionFollowingProcessor(),
        }

        outputs: List[Optional[str]] = [None] * len(items)

        for task_type, idxs in groups.items():
            sub_items = [items[i] for i in idxs]
            task_logic = logic_map.get(task_type, FactualQAProcessor())
            tokenizer = self.task_tokenizers.get(task_type, self.task_tokenizers["factual_qa"])

            # Preprocess with task-specific tokenizer
            prompts = task_logic.preprocess(sub_items, tokenizer)

            # Generate with task-specific model
            raw = self._generate_for_task(
                task_type, prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                batch_size=batch_size,
            )

            # Postprocess
            preds = task_logic.postprocess(raw, sub_items)

            for i, p in zip(idxs, preds):
                outputs[i] = p

        return [o if o is not None else "" for o in outputs]


class SelfConsistencyPipeline(RouterPipeline):
    """
    Pipeline that uses self-consistency for improved accuracy.
    Generates multiple samples per input and aggregates via majority voting
    using the SAME extraction logic as score.py for consistency.
    """
    def __init__(self, model_name: str, num_samples: int = 5):
        """
        Args:
            model_name: HuggingFace model identifier
            num_samples: Number of samples to generate per input for self-consistency
        """
        super().__init__(model_name)
        self.num_samples = num_samples

    @staticmethod
    def _extract_arc_answer(pred: str) -> str:
        """
        Extract ARC answer using EXACT same logic as score.py score_arc()
        """
        pred_clean = str(pred).strip().upper()
        # Same regex as score.py line 102
        match = re.search(r"\b([A-E])\b", pred_clean)
        if match:
            return match.group(1)
        else:
            # Same fallback as score.py line 106
            return pred_clean[0] if pred_clean and pred_clean[0].isalpha() else ""

    @staticmethod
    def _normalize_triviaqa_answer(pred: str) -> str:
        """
        Normalize TriviaQA answer using EXACT same logic as score.py _normalize_text()
        """
        import string

        # Same as score.py line 83: take first line
        pred = str(pred).strip().split("\n")[0]

        # Same normalization as score.py _normalize_text() (lines 29-33)
        ARTICLES = {"a", "an", "the"}
        PUNCT = set(string.punctuation)

        t = pred.strip().lower()
        t = " ".join(w for w in t.split() if w not in ARTICLES)
        t = "".join(ch for ch in t if ch not in PUNCT)
        return " ".join(t.split())

    def _aggregate_arc(self, samples: List[str]) -> str:
        """
        Aggregate ARC-C answers by majority vote on extracted letters.
        Uses the same extraction logic as score_arc().
        """
        letters = []
        for s in samples:
            letter = self._extract_arc_answer(s)
            if letter:
                letters.append(letter)

        if not letters:
            return samples[0] if samples else ""

        # Majority vote
        most_common = Counter(letters).most_common(1)[0][0]
        return most_common

    def _aggregate_triviaqa(self, samples: List[str]) -> str:
        """
        TODO: Implement self-consistency for TriviaQA.
        For now, just return the first sample (no aggregation).

        Future implementation should:
        - Normalize answers using same logic as score_triviaqa()
        - Use majority vote on normalized answers
        """
        return samples[0] if samples else ""

    def _aggregate_ifeval(self, samples: List[str]) -> str:
        """
        TODO: Implement self-consistency for IFEval.
        For now, just return the first sample (no aggregation).

        Future implementation could:
        - Use majority vote on exact matches
        - Fall back to longest response (more likely to satisfy constraints)
        """
        return samples[0] if samples else ""

    def run_with_router(
        self,
        items: List[Dict[str, Any]],
        router_fn,
        logic_map: Optional[Dict[str, ProcessLogic]] = None,
        batch_size: int = 16,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Override to implement self-consistency.
        Generates num_samples outputs per item and aggregates them.
        """
        if self.num_samples <= 1:
            # Fall back to standard behavior if num_samples is 1
            return super().run_with_router(
                items, router_fn, logic_map, batch_size,
                max_new_tokens, do_sample, temperature, top_p
            )

        # Force sampling for self-consistency
        do_sample = True

        # Replicate each item num_samples times
        expanded_items = []
        item_indices = []  # Track which original item each expanded item belongs to
        for idx, item in enumerate(items):
            for _ in range(self.num_samples):
                expanded_items.append(item)
                item_indices.append(idx)

        # Generate predictions for all expanded items
        expanded_outputs = super().run_with_router(
            expanded_items, router_fn, logic_map, batch_size,
            max_new_tokens, do_sample, temperature, top_p
        )

        # Group outputs by original item index
        grouped_outputs: Dict[int, List[str]] = {}
        for item_idx, output in zip(item_indices, expanded_outputs):
            if item_idx not in grouped_outputs:
                grouped_outputs[item_idx] = []
            grouped_outputs[item_idx].append(output)

        # Aggregate outputs for each original item
        final_outputs = []
        for idx, item in enumerate(items):
            samples = grouped_outputs.get(idx, [""])

            # Determine task type for aggregation
            task_type = router_fn(item)

            if task_type == "reasoning":  # ARC-C
                aggregated = self._aggregate_arc(samples)
            elif task_type == "factual_qa":  # TriviaQA
                aggregated = self._aggregate_triviaqa(samples)
            elif task_type == "instruction_following":  # IFEval
                aggregated = self._aggregate_ifeval(samples)
            else:
                # Default: take most common
                counts = Counter(s.strip() for s in samples if s.strip())
                aggregated = counts.most_common(1)[0][0] if counts else samples[0]

            final_outputs.append(aggregated)

        return final_outputs


class OurPipeline(MultiModelRouterPipeline):
    """
    Our pipeline implementation using MultiModelRouterPipeline.
    Routes questions to task-specific models configured in model_config.py.
    """
    def __init__(self, model_name: str = None):
        # model_name is ignored - uses MODEL_MAP from model_config.py
        super().__init__(model_name)