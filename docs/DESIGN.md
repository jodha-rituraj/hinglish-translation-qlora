# Design Notes

This page explains the main decisions in the pipeline, the reasons behind them, and what the first experiments taught.

## Problem

Hinglish is Hindi written in Latin script and mixed freely with English (*"Kal meeting ke baad call karta hoon"*). It is how a very large number of people actually text, but general-purpose LLMs tend to either answer in formal Devanagari Hindi or return the English unchanged. The goal is to teach a small open model to produce **natural, conversational Romanized Hinglish** from English input, cheaply enough to train on a single free GPU.

## Why QLoRA

| Approach | Trainable params (Phi-3-mini) | Approx. GPU memory to train | Fits a 16 GB T4? |
|---|---|---|---|
| Full fine-tuning (fp16 + Adam) | 3.8B (100%) | ~60 GB+ | No |
| LoRA on fp16 weights | 9.4M (0.25%) | ~10–12 GB | Barely |
| **QLoRA (4-bit NF4 base + LoRA)** | **9.4M (0.25%)** | **~5–7 GB** | **Yes, with headroom** |

*(Memory figures are rough estimates for short sequences.)*

QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) freezes the base model in 4-bit NF4 precision and trains small low-rank adapter matrices in higher precision. The pipeline uses:

- **NF4 + double quantization**, the configuration recommended in the paper, which quantizes the quantization constants as well.
- **bf16/fp16 compute dtype**, picked automatically: T4s don't support bf16, Ampere and newer GPUs do.
- **`prepare_model_for_kbit_training`**, which keeps layer norms in fp32 for stability and enables input gradients so gradient checkpointing works with frozen quantized weights.
- **Gradient checkpointing**, which recomputes activations in the backward pass to save memory.

The saved artifact is only the LoRA adapter (38 MB for Phi-3-mini, 18 MB for Llama-3.2-3B), loaded on top of the original base model at inference time.

## Base models

| Model | Params | LoRA targets | Why |
|---|---|---|---|
| `microsoft/Phi-3-mini-4k-instruct` (default) | 3.8B | `qkv_proj`, `o_proj` | Strong for its size, MIT licensed, not gated |
| `meta-llama/Llama-3.2-3B` | 3.2B | `q_proj`, `v_proj` | A different model family, for comparison; gated |

LoRA target modules depend on the architecture. Phi-3 fuses Q/K/V into one `qkv_proj` layer, while LLaMA and Gemma use separate `q_proj`/`k_proj`/`v_proj`. `config.get_lora_config()` picks the right set.

## Data

| Dataset | Pairs | Notes |
|---|---|---|
| [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish) | ~189k | Conversational: assistant commands, chit-chat, movie talk. **Used for training (first 20k pairs).** |
| [LingoIITGN/HinGE](https://huggingface.co/datasets/LingoIITGN/HinGE) | ~4.8k | Human-written Hinglish with several references per sentence; more formal. Used in an early pilot run. |

Both are saved in the same `{"input": english, "output": hinglish}` JSON format by `utils/prepare_training_data.py`, so either can be passed via `--data_path`.

## Training objective

Each example is formatted as:

```
Translate the following English sentence into natural, conversational Hinglish.

English: {english}
Hinglish: {hinglish}<eos>
```

The current code applies two details here. Both came out of the first experiments (see below):

1. **Loss masking on the prompt.** Prompt tokens get label `-100`, so they are ignored by the cross-entropy loss. The model is trained only to produce the translation, and every gradient step goes toward the actual task.
2. **An explicit EOS token after the answer.** Without it the model never learns where a translation ends. `DataCollatorForSeq2Seq` pads labels with `-100`, so reusing EOS as the pad token does not hide the real EOS from the loss.

## Hyperparameters

| Setting | Value | Rationale |
|---|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | Common default; alpha = 2r keeps the effective scale stable |
| Learning rate | 2e-4, linear decay, 30 warmup steps | Standard for QLoRA; adapters tolerate higher LRs than full fine-tuning |
| Effective batch size | 1 × 8 gradient accumulation = 8 | Fits T4 memory |
| Epochs | 2 (4,500 optimizer steps on 18k pairs) | Validation loss was still slowly improving at the end |
| Max sequence length | 256 tokens | Sentences in the data are well under this |
| Validation | 10% random split, evaluated every 200 steps; best checkpoint restored | |

## What the first experiments showed

The runs in `results/` were trained on a Colab T4 with an earlier version of this code. Reviewing their outputs (`results/*/sample_translations.csv`) surfaced three problems. All are fixed in the current code:

| Observation | Cause | Fix |
|---|---|---|
| Fine-tuned outputs often repeat themselves (`kya hai ? ? ? ? …`, `Please . Please . …`) | No EOS token in training targets, so the model never learned to stop | EOS appended to every target (`training/dataset.py`) |
| Part of the training signal was spent on the fixed instruction text | Labels were a copy of the full input, prompt included | Prompt tokens masked with `-100` |
| Some Llama outputs lose their first character (`ocal traffic kaisa hai?`) | Evaluation sliced the decoded text by `len(prompt)` characters, which drifts when tokenization changes whitespace | `infer.py` slices by **token** position instead |

The first two change how training behaves, so the reported losses come from the old objective: loss averaged over prompt and answer tokens. Those numbers are not directly comparable with a run of the current code.

## Limitations and next steps

- **No clean held-out test set yet.** The 100 sample translations came from the same 20k-pair pool used for training, so about 90% were seen during training. Next step: score the findnitai/HinGE **test** splits with chrF++ / BLEU.
- **Re-train with the fixed objective** (masking + EOS) and compare repetition rates against these runs.
- **Use the data more fully.** Train on more of the 189k findnitai pairs, and mix in HinGE for more formal registers.
- **Publish the adapters** to the Hugging Face Hub so others can try them without cloning the repo.
