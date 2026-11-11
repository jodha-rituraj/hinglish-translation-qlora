"""Summarize the fine-tuning runs in results/: loss curve + translation metrics.

Reads, for each run folder in results/:
    run_info.json             run settings and before/after validation loss
    trainer_state.json        HF Trainer log history (train / validation loss per step)
    sample_translations.csv   input, reference, base-model and fine-tuned outputs

Writes:
    results/validation_loss.png
    results/metrics.json

Needs no GPU or model weights:  python utils/summarize_results.py
"""

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sacrebleu.metrics import BLEU, CHRF  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUNS = {  # folder -> (display name, color)
    "phi3-mini": ("Phi-3-mini (3.8B)", "#2a78d6"),
    "llama-3.2-3b": ("Llama-3.2-3B", "#eb6834"),
}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def clean(output):
    """Keep the first non-empty line; models without EOS training ramble past the answer."""
    for line in str(output).splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def score(hypotheses, references):
    return {
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, [references]).score, 1),
        "bleu": round(BLEU().corpus_score(hypotheses, [references]).score, 1),
        "romanized_rate": round(sum(bool(h) and not DEVANAGARI.search(h) for h in hypotheses) / len(hypotheses), 2),
    }


def translation_metrics(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    refs = [r["actual translation"].strip() for r in rows]
    return {
        "n": len(rows),
        "base": score([clean(r["response"]) for r in rows], refs),
        "fine_tuned": score([clean(r["fine_tuned_response"]) for r in rows], refs),
    }


def eval_curve(run_dir):
    info = json.loads((run_dir / "run_info.json").read_text())
    state = json.loads((run_dir / "trainer_state.json").read_text())
    points = [(0, info["eval_loss_before"])]
    points += [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
    return points


def plot(curves, out_path):
    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name, (label, color) in RUNS.items():
        steps, losses = zip(*curves[name])
        ax.plot(steps, losses, color=color, linewidth=2, label=label, zorder=3)
        ax.scatter([steps[0], steps[-1]], [losses[0], losses[-1]], s=40, color=color,
                   edgecolors="#fcfcfb", linewidths=2, zorder=4)
        ax.annotate(f"{label}  {losses[-1]:.2f}", (steps[-1], losses[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink)

    ax.annotate("before fine-tuning", (0, max(c[0][1] for c in curves.values())), xytext=(8, 4),
                textcoords="offset points", fontsize=9, color=muted)
    ax.set_title("Validation loss during QLoRA fine-tuning", loc="left", fontsize=12, color=ink, pad=12)
    ax.set_xlabel("Optimizer step", fontsize=9, color=muted)
    ax.set_ylabel("Validation loss", fontsize=9, color=muted)
    ax.set_ylim(bottom=1.0)
    ax.set_xlim(right=max(s for c in curves.values() for s, _ in c) * 1.28)  # room for labels
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=ink)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved {out_path}")


def main():
    curves, metrics = {}, {}
    for name in RUNS:
        run_dir = RESULTS_DIR / name
        curves[name] = eval_curve(run_dir)
        info = json.loads((run_dir / "run_info.json").read_text())
        metrics[name] = {
            "eval_loss_before": info["eval_loss_before"],
            "eval_loss_after": info["eval_loss_after"],
            **translation_metrics(run_dir / "sample_translations.csv"),
        }

    plot(curves, RESULTS_DIR / "validation_loss.png")
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
