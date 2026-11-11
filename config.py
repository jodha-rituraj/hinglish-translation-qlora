"""Central configuration: base models, prompt format, hyperparameters, and paths.

Every value here is a default; most can be overridden from the command line
(see `python train.py --help`).
"""

from pathlib import Path

from peft import LoraConfig

# -------------------------------------------------------
# Base models
# -------------------------------------------------------
# Models this pipeline is set up for. Other causal LMs from the Hugging Face Hub
# should also work, but the LoRA target modules below may need adjusting.
SUPPORTED_MODELS = [
    # LLaMA family (gated: requires accepting the license on the HF Hub)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-2-7b-hf",

    # Phi family
    "microsoft/Phi-3-mini-4k-instruct",

    # Gemma family (gated)
    "google/gemma-2b",
    "google/gemma-2-9b",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
]

# Default model: 3.8B params, fits comfortably on a free 16 GB Colab T4 in 4-bit.
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


# -------------------------------------------------------
# Prompt format
# -------------------------------------------------------
# The model sees PROMPT_TEMPLATE and learns to continue it with the Hinglish
# translation followed by an EOS token. Only the translation part contributes
# to the training loss (see training/dataset.py).
PROMPT_TEMPLATE = (
    "Translate the following English sentence into natural, conversational Hinglish.\n\n"
    "English: {input}\n"
    "Hinglish:"
)


# -------------------------------------------------------
# Training hyperparameters (the values used for the runs in results/)
# -------------------------------------------------------
MAX_SAMPLES = 20000         # cap on training examples (0 on the CLI = use the full file)
MAX_SEQ_LENGTH = 256        # tokens; translations in HinGE are well under this
EVAL_SPLIT = 0.1            # fraction held out for validation loss
SEED = 42
BATCH_SIZE = 1              # per-device micro-batch
GRADIENT_ACC_STEPS = 8      # effective batch size = BATCH_SIZE * GRADIENT_ACC_STEPS
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_STEPS = 30
EVAL_STEPS = 200            # evaluate / checkpoint every N optimizer steps


# -------------------------------------------------------
# Paths (resolved relative to the repo, so scripts work from any directory)
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "eng_hinglish_train.json"  # findnitai/english-to-hinglish
OUTPUT_DIR = PROJECT_ROOT / "outputs"            # new training runs (git-ignored)
RESULTS_DIR = PROJECT_ROOT / "results"          # reported runs: adapters, logs, sample outputs


# -------------------------------------------------------
# LoRA
# -------------------------------------------------------
# Attention projection names differ between architectures: Phi-3 fuses Q/K/V
# into a single `qkv_proj`, while LLaMA and Gemma keep them separate.
LORA_TARGET_MODULES = {
    "phi": ["qkv_proj", "o_proj"],
    "default": ["q_proj", "v_proj"],
}


def get_lora_config(model_name: str) -> LoraConfig:
    """Return a fresh LoRA config with target modules matched to the model family."""
    family = "phi" if "phi" in model_name.lower() else "default"
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=LORA_TARGET_MODULES[family],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
