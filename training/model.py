"""Loading a base causal LM (optionally 4-/8-bit quantized) and its tokenizer."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

QUANTIZATION_CHOICES = ["4bit", "8bit", "none"]


def get_compute_dtype():
    """bf16 on GPUs that support it (Ampere+), fp16 on older GPUs (e.g. T4), fp32 on CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # Padded label positions are set to -100 by the collator, so reusing
        # EOS as the pad token does not stop the model from learning EOS.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, quantization="4bit", for_training=True):
    """Load `model_name` and its tokenizer.

    quantization:
        "4bit" - QLoRA setup: NF4 weights + double quantization (default)
        "8bit" - LLM.int8() weights
        "none" - full-precision weights (needed on CPU / Apple Silicon)
    """
    if quantization not in QUANTIZATION_CHOICES:
        raise ValueError(f"quantization must be one of {QUANTIZATION_CHOICES}, got {quantization!r}")

    if quantization != "none" and not torch.cuda.is_available():
        raise RuntimeError(
            f"{quantization} quantization requires a CUDA GPU (bitsandbytes). "
            "Use `--quantization none` to run on CPU."
        )

    print(f"Loading model: {model_name} (quantization={quantization})")
    tokenizer = load_tokenizer(model_name)
    compute_dtype = get_compute_dtype()

    if quantization == "none":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=compute_dtype,
            device_map="auto",
        )
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=quantization == "4bit",
            load_in_8bit=quantization == "8bit",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )

        if for_training:
            # Casts layer norms to fp32 and enables input gradients so that
            # gradient checkpointing works with frozen quantized weights.
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if for_training:
        model.config.use_cache = False  # KV cache is useless (and warns) during training

    return model, tokenizer
