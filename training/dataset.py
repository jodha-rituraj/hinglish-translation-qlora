"""Loading, formatting, and tokenizing English -> Hinglish training pairs."""

import json
from pathlib import Path

from config import PROMPT_TEMPLATE

IGNORE_INDEX = -100  # label value ignored by PyTorch's cross-entropy loss


def load_training_data(data_path, max_samples=None):
    """Load a JSON list of {"input": <english>, "output": <hinglish>} records."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found at: {data_path}. "
            "Run `python utils/prepare_training_data.py` to download it."
        )

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    return data


def build_prompt(english_text, prompt_template=PROMPT_TEMPLATE):
    """Format an English sentence into the instruction prompt (without the answer)."""
    return prompt_template.format(input=english_text.strip())


def prepare_training_data(raw_data, prompt_template=PROMPT_TEMPLATE):
    """Turn raw pairs into {"prompt", "response"} records, dropping empty ones."""
    training_data = []

    for item in raw_data:
        input_text = item.get("input", "").strip()
        output_text = item.get("output", "").strip()
        if not input_text or not output_text:
            continue

        training_data.append({
            "prompt": build_prompt(input_text, prompt_template),
            "response": output_text,
        })

    return training_data


def tokenize_sample(sample, tokenizer, max_length):
    """Tokenize one prompt/response pair for causal-LM fine-tuning.

    - An EOS token is appended so the model learns where a translation ends
      (otherwise it keeps generating past the answer at inference time).
    - Prompt tokens are masked with IGNORE_INDEX in `labels`, so the loss is
      computed only on the Hinglish translation, not on the fixed instruction.
    """
    prompt_ids = tokenizer(sample["prompt"])["input_ids"]
    full_ids = tokenizer(sample["prompt"] + " " + sample["response"])["input_ids"]
    if not full_ids or full_ids[-1] != tokenizer.eos_token_id:
        full_ids = full_ids + [tokenizer.eos_token_id]

    # Tokenizing the prompt alone and the full text can very occasionally
    # disagree by a token at the boundary; masking len(prompt_ids) is a safe
    # approximation that is standard practice.
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]

    full_ids = full_ids[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
