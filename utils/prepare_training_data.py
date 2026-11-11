"""Download English -> Hinglish datasets from the Hugging Face Hub into data/*.json.

Each output file is a JSON list of {"input": <english>, "output": <hinglish>}.
When a dataset provides several Hinglish references for one English sentence,
each reference becomes its own training pair.

Usage:
    python utils/prepare_training_data.py                  # both datasets
    python utils/prepare_training_data.py --dataset findnitai  # only the default training set
"""

import argparse
import ast
import json
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_references(value):
    """HinGE stores multiple references as a stringified Python list; others are plain strings."""
    value = value.strip()
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value

    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    return [str(parsed)]


def save_json(rows, out_path, english_col, hinglish_col):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    for row in rows:
        english = row[english_col].strip()
        for hinglish in parse_references(row[hinglish_col]):
            items.append({"input": english, "output": hinglish.strip()})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(items)} pairs to {out_path}")


def download_hinge():
    """LingoIITGN/HinGE: English sentences with human-written Hinglish translations."""
    dataset = load_dataset("LingoIITGN/HinGE")
    for split in ("train", "validation", "test"):
        if split in dataset:
            save_json(dataset[split], DATA_DIR / f"hinge_{split}.json",
                      english_col="English", hinglish_col="Human-generated Hinglish")


def download_findnitai():
    """findnitai/english-to-hinglish: ~189k conversational pairs (noisier, larger)."""
    dataset = load_dataset("findnitai/english-to-hinglish")
    for split in ("train", "validation", "test"):
        if split in dataset:
            save_json(dataset[split]["translation"], DATA_DIR / f"eng_hinglish_{split}.json",
                      english_col="en", hinglish_col="hi_ng")


DATASETS = {"hinge": download_hinge, "findnitai": download_findnitai}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = parser.parse_args()

    for name, download in DATASETS.items():
        if args.dataset in (name, "all"):
            print(f"Downloading {name}...")
            download()
