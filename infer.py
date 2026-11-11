"""Translate English sentences into Hinglish with a fine-tuned LoRA adapter.

Examples:
    python infer.py --text "Where are you going tonight?"
    python infer.py --compare --text "I will call you after the meeting."
    echo "What's the name of the movie?" | python infer.py
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel

import config
from training.dataset import build_prompt
from training.model import QUANTIZATION_CHOICES, load_model

DEFAULT_ADAPTER = config.OUTPUT_DIR / "adapter"


def parse_args():
    parser = argparse.ArgumentParser(
        description="English -> Hinglish translation with a fine-tuned LoRA adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adapter_path", type=Path, default=DEFAULT_ADAPTER, help="Directory produced by train.py")
    parser.add_argument("--base_model", type=str, default=None, help="Base model (default: read from the adapter config)")
    parser.add_argument("--quantization", choices=QUANTIZATION_CHOICES,
                        default="4bit" if torch.cuda.is_available() else "none", help="Base-model weight quantization")
    parser.add_argument("--text", action="append", help="English sentence to translate (repeatable). Reads stdin if omitted.")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--compare", action="store_true", help="Also show the base model's output (adapter disabled)")
    return parser.parse_args()


def resolve_base_model(adapter_path, base_model):
    if base_model:
        return base_model
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        sys.exit(f"No adapter found at {adapter_path}. Train one with `python train.py` "
                 "or pass --adapter_path.")
    return json.loads(adapter_config.read_text())["base_model_name_or_path"]


@torch.no_grad()
def translate(model, tokenizer, english, max_new_tokens):
    inputs = tokenizer(build_prompt(english), return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: deterministic, good for translation
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text.split("\n")[0].strip()  # the base model may ramble past the answer


def main():
    args = parse_args()
    base_model = resolve_base_model(args.adapter_path, args.base_model)

    model, tokenizer = load_model(base_model, quantization=args.quantization, for_training=False)
    model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    sentences = args.text or [line.strip() for line in sys.stdin if line.strip()]
    for english in sentences:
        print(f"\nEnglish:    {english}")
        if args.compare:
            with model.disable_adapter():
                print(f"Base model: {translate(model, tokenizer, english, args.max_new_tokens)}")
        print(f"Fine-tuned: {translate(model, tokenizer, english, args.max_new_tokens)}")


if __name__ == "__main__":
    main()
