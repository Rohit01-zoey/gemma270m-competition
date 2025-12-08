import argparse
import numpy as np
import json
import os
from pipelines import BasePipeline, FactualQAProcessor, ReasoningProcessor, apply_instruct_template
from generate import generate_rows, _write_jsonl, _read_jsonl
from score import score_from_rows
import pickle
from rank_bm25 import BM25Okapi
from datasets import load_dataset
import re
from nltk.stem import PorterStemmer

# Initialize stemmer globally
stemmer = PorterStemmer()

def tokenize_for_bm25(text):
    # Remove punctuation, keep only alphanumeric and whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split and stem
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]
    #return [stemmer.stem(token) for token in tokens if len(token) > 1]


def load_msmarco(num_passages: int = None):
    cache_suffix = "full" if num_passages is None else str(num_passages)
    # IMPORTANT: Update cache filename to reflect new tokenization
    cache_file = f"/home/users/ms1254/gemma270m-competition/msmarco_{cache_suffix}_bm25_nostem.pkl"
    
    if os.path.exists(cache_file):
        print(f"Loading MS MARCO from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    if num_passages is None:
        print("Loading FULL MS MARCO dataset (all passages)...")
    else:
        print(f"Loading {num_passages:,} MS MARCO passages...")
    
    dataset = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    
    corpus = []
    tokenized = []
    count = 0
    
    for i, item in enumerate(dataset):
        if num_passages is not None and count >= num_passages:
            break
        
        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} queries, {count:,} passages...")
        
        # Access passages correctly based on the dataset structure
        passages_dict = item['passages']
        passage_texts = passages_dict['passage_text']  # This is a list of strings
        
        # Add each passage to corpus
        for passage_text in passage_texts:
            if passage_text and passage_text.strip():
                corpus.append(passage_text)
                # Use improved tokenization
                tokenized.append(tokenize_for_bm25(passage_text))
                count += 1
                
                if num_passages is not None and count >= num_passages:
                    break
    
    print(f"Building BM25 index on {len(corpus):,} passages...")
    bm25 = BM25Okapi(tokenized)
    
    data = {'corpus': corpus, 'bm25': bm25}
    
    print(f"Saving index to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Indexed {len(corpus):,} MS MARCO passages (with stemming)")
    return data


class RAGPipeline(BasePipeline):
    def __init__(self, model_name: str, top_k: int = 3, num_passages: int = None):
        super().__init__(model_name)
        self.top_k = top_k
        self.max_context_chars = 4000
        
        # Load MS MARCO
        data = load_msmarco(num_passages)
        self.corpus = data['corpus']
        self.bm25 = data['bm25']
        
        # ADDITION: Store retrieval info for inspection
        self.retrieval_cache = {}

    def retrieve(self, query: str) -> str:
        query = query.strip()
        if not query:
            return ""
        # Use improved tokenization (same as indexing)
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        passages = [self.corpus[i] for i in top_idx]
        context = "\n\n".join(passages)

        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars]

        return context

    def retrieve_with_indices(self, query: str):
        query = query.strip()
        if not query:
            return {"context": "", "indices": [], "scores": []}
        # Use improved tokenization (same as indexing)
        tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        passages = [self.corpus[i] for i in top_idx]
        context = "\n\n".join(passages)
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars]
        return {
            "context": context, 
            "indices": top_idx.tolist(),
            "scores": scores[top_idx].tolist(),
            "passages": passages,  # ADDITION: Include actual passage texts
            "query_tokens": tokens  # ADDITION: Include tokenized query
        }


class RAGFactualQAProcessor(FactualQAProcessor):
    def __init__(self, rag_pipeline):
        super().__init__(few_shot=0)
        self.rag = rag_pipeline
    
    def preprocess(self, items, tokenizer):
        prompts = []
        for it in items:
            q = it.get("question") or it.get("query") or ""
            item_id = it.get("id", q)  # Use id if available, else question as key
            
            # MODIFICATION: Use retrieve_with_indices to get all info
            retrieval_info = self.rag.retrieve_with_indices(q)
            
            # ADDITION: Store retrieval info for later inspection
            self.rag.retrieval_cache[item_id] = retrieval_info
            
            context = retrieval_info["context"]
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on the provided context if it's relevant. If the context doesn't contain relevant information, ignore it and answer based on your own knowledge. Keep your answer concise."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"}
            ]
            prompts.append(apply_instruct_template(messages, tokenizer))
        return prompts


class RAGReasoningProcessor(ReasoningProcessor):
    def __init__(self, rag_pipeline):
        super().__init__()
        self.rag = rag_pipeline
    
    def preprocess(self, items, tokenizer):
        prompts = []
        for it in items:
            q = it.get("question", "")
            item_id = it.get("id", q)  # Use id if available, else question as key
            
            # MODIFICATION: Use retrieve_with_indices to get all info
            retrieval_info = self.rag.retrieve_with_indices(q)
            
            # ADDITION: Store retrieval info for later inspection
            self.rag.retrieval_cache[item_id] = retrieval_info
            
            context = retrieval_info["context"]
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Use the provided context if it's relevant to answering the question. If the context is not helpful, rely on your own knowledge. Choose the best answer from the given options."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"}
            ]
            prompts.append(apply_instruct_template(messages, tokenizer))
        return prompts


def __default_data_file_for_task(task: str) -> str:
    if task == "triviaqa":
        return "data/triviaqa_test.jsonl"
    if task == "arc-c":
        return "data/arc_c_test.jsonl"
    raise ValueError(f"Unknown task: {task}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["triviaqa", "arc-c"], default="triviaqa")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data-size", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve")
    ap.add_argument("--num-passages", type=int, default=None, 
                    help="Total MS MARCO passages to index (None = use all)")
    ap.add_argument("--use-full-msmarco", action="store_true",
                    help="Use full MS MARCO dataset (all passages)")
    ap.add_argument("--output-prefix", type=str, default=None,
                    help="Custom prefix for output files (overrides default naming)")

    args = ap.parse_args()

    # If --use-full-msmarco flag is set, override num_passages
    if args.use_full_msmarco:
        args.num_passages = None
        print("Using FULL MS MARCO dataset")
    elif args.num_passages is None:
        args.num_passages = 100000  # Default to 100k if not specified
        print(f"Using {args.num_passages:,} MS MARCO passages (default)")
    else:
        print(f"Using {args.num_passages:,} MS MARCO passages")

    data_file = __default_data_file_for_task(args.task)
    items = _read_jsonl(data_file)[:args.data_size]

    os.makedirs(args.out_dir, exist_ok=True)
    
    # MODIFIED: Use custom prefix if provided, otherwise auto-generate
    if args.output_prefix:
        filename = args.output_prefix
        print(f"Using custom output prefix: {filename}")
    else:
        corpus_label = "full" if args.num_passages is None else str(args.num_passages)
        filename = f"{args.task}_rag_msmarco_{corpus_label}_n{args.data_size}_k{args.top_k}_stemmed_base"
        print(f"Using auto-generated filename: {filename}")
    
    preds_path = f"{args.out_dir}/{filename}_preds.jsonl"
    metrics_path = f"{args.out_dir}/{filename}_metrics.json"

    # Create RAG pipeline with MS MARCO
    pipeline = RAGPipeline(args.model, top_k=args.top_k, num_passages=args.num_passages)

    # Select appropriate processor
    if args.task == "triviaqa":
        logic = RAGFactualQAProcessor(pipeline)
    elif args.task == "arc-c":
        logic = RAGReasoningProcessor(pipeline)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    
    # Generate predictions
    rows = generate_rows(items, pipeline, batch_size=args.batch_size, 
                        max_new_tokens=args.max_new_tokens, logic=logic)
    
    # ADDITION: Merge retrieval info into rows
    for row in rows:
        item_id = row.get("id", row.get("question"))
        if item_id in pipeline.retrieval_cache:
            row["retrieval_info"] = pipeline.retrieval_cache[item_id]
    
    # Save results
    _write_jsonl(preds_path, rows)
    result = score_from_rows(args.task, rows)
    
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({"task": args.task, "n": len(rows), **result}, indent=2))


if __name__ == "__main__":
    main()