# English → Hinglish Translation with QLoRA

Fine-tuning a small open LLM (**Phi-3-mini, 3.8B**) to translate English into natural, conversational **Hinglish** (Hindi written in Latin script, mixed with English), using **QLoRA**. The whole pipeline runs on a single free Colab T4 GPU.

```
English:  It was presented to the Legislative Council in 1856 and was passed in 1860.
Hinglish: Ise 1856 mein legislative council ke samaksh prastut kiya gya and 1860 mein paarit kiya gya.
```
<sub>Example pair from the HinGE training data.</sub>

## Highlights

- **QLoRA end to end:** 4-bit NF4 base model with double quantization, k-bit training preparation, gradient checkpointing, and LoRA adapters on the attention projections. Only **~0.25% of parameters (9.4M)** are trained.
- **Correct causal-LM objective:** the loss is computed **only on the Hinglish answer** (prompt tokens are masked), and an explicit EOS token teaches the model where a translation ends.
- **Works across model families:** Phi-3, LLaMA 3.x and Gemma, with LoRA target modules chosen per architecture.
- **Reproducible:** fixed seeds, all settings in one `config.py`, and the run config plus loss history are saved with every run.
- **One-command pipeline:** `scripts/run_pipeline.sh` installs dependencies, downloads data, trains, and prints base-model vs fine-tuned translations side by side.

See **[docs/DESIGN.md](docs/DESIGN.md)** for the reasoning behind each choice (memory budget, objective, hyperparameters, limitations).

## How it works

```
 HF Hub datasets ──► utils/prepare_training_data.py ──► data/*.json  {"input": en, "output": hinglish}
                                                            │
                                                            ▼
                       training/dataset.py   prompt template + EOS, mask prompt tokens in labels
                                                            │
 Phi-3-mini (4-bit NF4) ◄── training/model.py               ▼
          │                                    train.py  (HF Trainer + PEFT LoRA)
          └──────────────────────────────────────────►  outputs/adapter/  (LoRA weights, ~40 MB)
                                                            │
                                                            ▼
                                                 infer.py  base model + adapter → Hinglish
```

## Project structure

```
├── config.py                     # models, prompt template, hyperparameters, LoRA config, paths
├── train.py                      # fine-tuning entry point (CLI)
├── infer.py                      # translate with a trained adapter; --compare vs base model
├── training/
│   ├── dataset.py                # loading, prompt formatting, tokenization + loss masking
│   └── model.py                  # base model / tokenizer loading with 4-bit or 8-bit quantization
├── utils/
│   └── prepare_training_data.py  # downloads HinGE + findnitai datasets into data/
├── scripts/
│   └── run_pipeline.sh           # install → data → train → sample translations
├── data/                         # prepared JSON datasets
└── docs/DESIGN.md                # design decisions and next steps
```

## Setup

**Requirements:** Python 3.10+. Training with 4-bit quantization needs an NVIDIA GPU with CUDA (a free Colab T4 with 16 GB is enough). CPU and Apple Silicon can run the smoke test with `--quantization none`.

```bash
git clone https://github.com/jodha-rituraj/hinglish-translation-qlora.git
cd hinglish-translation-qlora

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The default model (`microsoft/Phi-3-mini-4k-instruct`) is not gated. To use LLaMA or Gemma, first accept the model license on the Hugging Face Hub, then run `huggingface-cli login`.

## Usage

### Option A: one command

```bash
bash scripts/run_pipeline.sh                    # full run with defaults
bash scripts/run_pipeline.sh --num_epochs 1     # any train.py flag can be appended
```

**On Google Colab** (Runtime → Change runtime type → T4 GPU):

```python
!git clone https://github.com/jodha-rituraj/hinglish-translation-qlora.git
%cd hinglish-translation-qlora
!bash scripts/run_pipeline.sh
```

### Option B: step by step

**1. Prepare data** (`data/hinge_train.json` is already included; this regenerates it):

```bash
python utils/prepare_training_data.py --dataset hinge   # or: findnitai | all
```

**2. Train:**

```bash
python train.py                                  # QLoRA on Phi-3-mini, 1,000 samples, 2 epochs
python train.py --max_samples 0 --num_epochs 3   # use the full dataset
python train.py --model_name meta-llama/Llama-3.2-3B
python train.py --help                           # all options
```

To check that the pipeline works on a laptop (CPU / Apple Silicon, 5 steps, no quantization):

```bash
python train.py --smoke_test --quantization none --model_name HuggingFaceTB/SmolLM2-135M
```

**3. Translate:**

```bash
python infer.py --text "Where are you going tonight?"
python infer.py --compare --text "I will call you after the meeting."   # base vs fine-tuned
echo "What's the name of the movie?" | python infer.py
```

### Outputs

```
outputs/
├── adapter/            # final LoRA adapter + tokenizer (load with infer.py)
├── checkpoints/        # intermediate checkpoints (best 2 kept)
├── log_history.json    # training / validation loss per logging step
└── run_config.json     # exact arguments used for the run
```

## Results

> 🚧 **Pending a full GPU training run.** The pipeline has been verified end to end on CPU with a tiny randomly-initialized Phi-3-architecture model (tokenization and loss masking, training loop, adapter saving, inference), but full-run metrics haven't been recorded yet. After running `scripts/run_pipeline.sh` on a GPU, fill this section in from `outputs/log_history.json` and the `--compare` output.

| Model | Train loss | Validation loss |
|---|---|---|
| Phi-3-mini + QLoRA (1k samples, 2 epochs) | – | – |

| English | Base Phi-3-mini | Fine-tuned |
|---|---|---|
| Where are you going tonight? | – | – |
| I will call you after the meeting. | – | – |

## Configuration

All defaults are in [`config.py`](config.py) and most can be overridden from the CLI:

| Setting | Default |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` |
| Quantization | 4-bit NF4, double quantization |
| LoRA | r=16, α=32, dropout=0.05 |
| Learning rate | 2e-4, linear decay, 30 warmup steps |
| Effective batch size | 8 (1 × 8 gradient accumulation) |
| Epochs / samples | 2 / 1,000 |
| Validation split | 10% |

## Datasets

- [LingoIITGN/HinGE](https://huggingface.co/datasets/LingoIITGN/HinGE): human-written Hinglish translations (used by default)
- [findnitai/english-to-hinglish](https://huggingface.co/datasets/findnitai/english-to-hinglish): ~189k conversational pairs

Datasets and base models are governed by their own licenses.

## Next steps

- Evaluate on the HinGE test split with chrF++ / BLEU (base vs fine-tuned)
- Train/validation split by unique English sentence to avoid reference leakage
- Publish the trained adapter to the Hugging Face Hub

## License

Code is released under the [MIT License](LICENSE).
