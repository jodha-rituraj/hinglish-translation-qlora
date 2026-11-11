# Design Notes

This page explains the main decisions in the pipeline and the reasons behind them.

## Problem

Hinglish is Hindi written in Latin script and mixed freely with English (*"Kal meeting ke baad call karta hoon"*). It is how a very large number of people actually text, but general-purpose LLMs tend to either answer in formal Devanagari Hindi or return the English unchanged. The goal is to teach a small open model to produce **natural, conversational Hinglish** from English input, cheaply enough to train on a single free GPU.

## Why QLoRA

| Approach | Trainable params (Phi-3-mini) | Approx. GPU memory to train | Fits a 16 GB T4? |
|---|---|---|---|
| Full fine-tuning (fp16 + Adam) | 3.8B (100%) | ~60 GB+ | No |
| LoRA on fp16 weights | 9.4M (0.25%) | ~10–12 GB | Barely |
| **QLoRA (4-bit NF4 base + LoRA)** | **9.4M (0.25%)** | **~5–7 GB** | **Yes, with headroom** |

*(Memory figures are rough estimates for short sequences with gradient checkpointing.)*

QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) freezes the base model in 4-bit NF4 precision and trains small low-rank adapter matrices in higher precision. The pipeline uses:

- **NF4 + double quantization**, the configuration recommended in the paper, which quantizes the quantization constants as well.
- **bf16/fp16 compute dtype**, picked automatically: T4s don't support bf16, Ampere and newer GPUs do.
- **`prepare_model_for_kbit_training`**, which keeps layer norms in fp32 for stability and enables input gradients so gradient checkpointing works with frozen quantized weights.
- **Gradient checkpointing**, which recomputes activations in the backward pass to save memory.

The saved artifact is only the LoRA adapter, a few tens of MB rather than ~7.6 GB, and it is loaded on top of the original base model at inference time.

## Base model

The default is **`microsoft/Phi-3-mini-4k-instruct`** (3.8B, MIT licensed, not gated). It is strong for its size, already follows instructions, and fits a T4 in 4-bit. Several LLaMA and Gemma models are listed in `config.SUPPORTED_MODELS` as drop-in alternatives (`--model_name`).

LoRA target modules depend on the architecture. Phi-3 fuses Q/K/V into one `qkv_proj` layer, while LLaMA and Gemma use separate `q_proj`/`k_proj`/`v_proj`. `config.get_lora_config()` picks the right set.

## Data

| Dataset | Pairs | Notes |
|---|---|---|
| [LingoIITGN/HinGE](https://huggingface.co/datasets/LingoIITGN/HinGE) | ~4.8k | Human-written Hinglish; several references per sentence. **Used for training by default.** |
| [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | ~189k | Larger, conversational, noisier. Downloaded but not used by default. |

When HinGE has several human references for one English sentence, each becomes its own training pair. This teaches the model that more than one valid Hinglish rendering exists.

> **Caveat:** because of that expansion, a random train/validation split can put different references for the *same* English sentence on both sides. Validation loss is therefore somewhat optimistic. Splitting by unique English sentence would fix this (see next steps).

## Training objective

Each example is formatted as:

```
Translate the following English sentence into natural, conversational Hinglish.

English: {english}
Hinglish: {hinglish}<eos>
```

Two details matter here:

1. **Loss masking on the prompt.** Prompt tokens get label `-100`, so they are ignored by the cross-entropy loss. The model is trained only to produce the translation, not to memorize the fixed instruction, and every gradient step goes toward the actual task.
2. **An explicit EOS token after the answer.** Without it the model never learns where a translation ends and keeps generating after the answer. `DataCollatorForSeq2Seq` pads labels with `-100`, so reusing EOS as the pad token does not hide the real EOS from the loss.

## Hyperparameters

| Setting | Value | Rationale |
|---|---|---|
| LoRA rank / alpha | 16 / 32 | Common default; alpha = 2r keeps the effective scale stable |
| LoRA dropout | 0.05 | Light regularization for a small dataset |
| Learning rate | 2e-4, linear decay, 30 warmup steps | Standard for QLoRA; adapters tolerate higher LRs than full fine-tuning |
| Effective batch size | 1 × 8 grad-accum = 8 | Fits T4 memory |
| Epochs | 2 | Small dataset; validation loss is tracked to catch overfitting |
| Max sequence length | 256 tokens | HinGE sentences are well under this |
| Samples | 1,000 (`--max_samples 0` for all) | Fast iteration on a free GPU |

The best checkpoint by validation loss is restored at the end of training (`load_best_model_at_end`).

## Limitations and next steps

- **No automatic translation metrics yet.** Only training and validation loss are tracked. Next step: score the HinGE test split with chrF++ / BLEU (e.g. `sacrebleu`), comparing the base model with the fine-tuned one.
- **Leakage-free split.** Split by unique English sentence rather than by pair.
- **Use the full data.** Train on all of HinGE, and try mixing in a filtered subset of the larger findnitai corpus.
- **Publish the adapter** to the Hugging Face Hub so others can try it without training.
