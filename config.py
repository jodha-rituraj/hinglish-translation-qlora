from pathlib import Path

# -------------------------------------------------------
# Allowed base models for QLoRA fine-tuning
# -------------------------------------------------------
ALLOWED_MODELS = [
    # LLaMA family
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-7B",
    "meta-llama/Llama-2-7b-hf",

    # Phi family
    "microsoft/Phi-3-mini-4k-instruct",

    # Gemma models
    "google/gemma-2b",
    "google/gemma-2-9b",
    "google/gemma-1.1-2b",
    "google/gemma-1.1-7b"
]

# Default model
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


# -------------------------------------------------------
# Prompt template for the task
# -------------------------------------------------------
TRANSLATION_PROMPT = (
    "Translate the following English sentence into natural, conversational Hinglish.\n\n"
    "English: {input}\n"
    "Hinglish: {output}"
)


# -------------------------------------------------------
# Training hyperparameters
# -------------------------------------------------------
MAX_SAMPLES = 1000
SEED = 42
BATCH_SIZE = 2
GRADIENT_ACC_STEPS = 16
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4


# -------------------------------------------------------
# Project directories (generic, no JSON paths)
# -------------------------------------------------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"