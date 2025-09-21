import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from transformers import AutoTokenizer


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)


DATASETS_INFO: Dict[str, Dict] = {
    # name                        split    key mapping or special handling
    "argilla/dpo-mix-7k":          {"split": "train"},
    "joecodecreations/alpaca_data_52k": {"split": "train"},
    "HuggingFaceH4/mt_bench_prompts":   {"split": "train"},
    "lmsys/mt_bench_human_judgments":   {"split": "train"},
    "truthfulqa/truthful_qa":      {"split": "generation"},
    "cais/mmlu":                  {"split": "all"},
    "allenai/real-toxicity-prompts":  {"split": "train"},
    # NOTE: ShareGPT-clean-20k and HH-RLHF require gated access:
    "HuggingFaceH4/ShareGPT_cleaned":   {"split": "train"},
    "Anthropic/hh-rlhf":                {"split": "train"},
}


def _env_check():
    if os.getenv("HF_TOKEN") is None:
        print(
            "WARNING: environment variable HF_TOKEN is not set. "
            "If any gated models/datasets are requested, download will fail."
        )


def _load_one(name: str, split: str, streaming: bool = False):
    try:
        return load_dataset(name, split=split, streaming=streaming, use_auth_token=os.getenv("HF_TOKEN"))
    except Exception as e:
        print(f"WARNING: Failed to load dataset '{name}': {e}")
        print(f"Skipping dataset '{name}' and continuing with available datasets.")
        return None


def load_all_raw(streaming: bool = False) -> Dict[str, Dataset]:
    """
    Downloads every dataset declared in DATASETS_INFO to `data/`
    and returns a mapping of dataset-name -> HF dataset object.
    """
    _env_check()
    loaded = {}
    for name, meta in DATASETS_INFO.items():
        print(f"Downloading {name} …")
        ds = _load_one(name, split=meta["split"], streaming=streaming)
        if ds is not None:
            loaded[name] = ds
    return loaded


# ---------------------------  Unified Mixture  --------------------------- #
PROMPT_COLUMN_CANDIDATES = ["prompt", "instruction", "question", "query", "context"]
RESPONSE_COLUMN_CANDIDATES = [
    "chosen",
    "answer",
    "response",
    "output",
    "completion",
    "text",
]


def _detect_col(example: Dict, candidates: List[str]) -> str:
    for c in candidates:
        if c in example and example[c] is not None:
            return c
    raise KeyError(f"Could not detect any of {candidates} in keys: {list(example.keys())}")


def _process_example(example: Dict) -> Dict[str, str]:
    if "chosen" in example and "rejected" in example:
        if "prompt" in example:
            prompt = example["prompt"]
        elif "instruction" in example:
            prompt = example["instruction"]
        else:
            chosen_text = str(example["chosen"])
            if len(chosen_text) > 100:
                prompt = chosen_text[:100] + "..."
            else:
                prompt = chosen_text
        return {"prompt": prompt, "response": str(example["chosen"])}
    else:
        prompt_col = _detect_col(example, PROMPT_COLUMN_CANDIDATES)
        resp_col = _detect_col(example, RESPONSE_COLUMN_CANDIDATES)
        return {"prompt": example[prompt_col], "response": example[resp_col]}


def build_mixture(
    raw_datasets: Dict[str, Dataset], mixture_ratios: Dict[str, int], smoke: bool = False
) -> Dataset:
    """
    Construct the mixed training dataset with the desired sampling ratios.
    """
    processed_splits: List[Dataset] = []
    for name, ratio in mixture_ratios.items():
        if name not in raw_datasets:
            print(f"WARNING: Dataset '{name}' not available, skipping...")
            continue
        raw = raw_datasets[name]
        # Map and optionally sub-sample (for smoke test)
        if isinstance(raw, Dataset):
            mapped = raw.map(_process_example)
        else:  # streaming
            mapped = raw.map(_process_example)
        if smoke:
            mapped = mapped.shuffle(seed=42).select(range(min(10, len(mapped))))
        processed_splits.extend([mapped] * ratio)
    
    if not processed_splits:
        raise RuntimeError("No datasets were successfully loaded. Cannot proceed with training.")
    
    final_dataset = concatenate_datasets(processed_splits).shuffle(seed=13)
    return final_dataset


# ---------------------------  Tokenisation  --------------------------- #
def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 128,
    smoke: bool = False,
) -> Dataset:
    def _tok_fn(example):
        joined = f"### Prompt:\n{example['prompt']}\n### Response:\n{example['response']}"
        tok = tokenizer(
            joined,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tok["labels"] = tok["input_ids"].copy()
        return tok

    return dataset.map(
        _tok_fn,
        batched=False,
        remove_columns=["prompt", "response"],
        num_proc=1,  # Always use single process to avoid file handle issues
    )
