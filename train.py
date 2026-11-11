"""Fine-tune a causal LM to translate English into Hinglish with QLoRA.

Examples:
    # Full QLoRA run on a CUDA GPU (e.g. a free Colab T4)
    python train.py

    # Quick pipeline check on CPU / Apple Silicon: 5 steps, no quantization
    python train.py --smoke_test --quantization none --model_name <small-model>
"""

import argparse
import json
from pathlib import Path

import datasets
import torch
from peft import get_peft_model
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

import config
from training.dataset import load_training_data, prepare_training_data, tokenize_sample
from training.model import QUANTIZATION_CHOICES, get_compute_dtype, load_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an LLM for English -> Hinglish translation using QLoRA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="Base model on the HF Hub (or a local path)")
    parser.add_argument("--data_path", type=Path, default=config.DEFAULT_DATA_PATH, help="Training data JSON file")
    parser.add_argument("--output_dir", type=Path, default=config.OUTPUT_DIR, help="Where checkpoints, logs, and the final adapter are written")
    parser.add_argument("--quantization", choices=QUANTIZATION_CHOICES, default="4bit", help="Base-model weight quantization (4bit = QLoRA)")
    parser.add_argument("--max_samples", type=int, default=config.MAX_SAMPLES, help="Max training examples to use (0 = all)")
    parser.add_argument("--max_seq_length", type=int, default=config.MAX_SEQ_LENGTH, help="Max tokens per example")
    parser.add_argument("--eval_split", type=float, default=config.EVAL_SPLIT, help="Fraction of data held out for validation (0 disables eval)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Per-device micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=config.GRADIENT_ACC_STEPS, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=float, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=config.LEARNING_RATE, help="Peak learning rate")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--smoke_test", action="store_true", help="Run 5 steps on 32 samples to check the pipeline end to end")

    return parser.parse_args()


def build_datasets(args, tokenizer):
    max_samples = args.max_samples or None
    if args.smoke_test:
        max_samples = 32

    raw_data = load_training_data(args.data_path, max_samples=max_samples)
    records = prepare_training_data(raw_data)
    print(f"Loaded {len(records)} training examples from {args.data_path}")

    dataset = datasets.Dataset.from_list(records)
    dataset = dataset.map(
        lambda sample: tokenize_sample(sample, tokenizer, args.max_seq_length),
        remove_columns=["prompt", "response"],
    )

    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        return split["train"], split["test"]
    return dataset, None


def build_training_args(args, has_eval):
    use_cuda = torch.cuda.is_available()
    compute_dtype = get_compute_dtype()

    common = dict(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        gradient_checkpointing=True,  # trades compute for a large VRAM saving
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
    )

    if args.smoke_test:
        return TrainingArguments(
            **common,
            max_steps=5,
            logging_steps=1,
            eval_strategy="no",
            save_strategy="no",
        )

    return TrainingArguments(
        **common,
        num_train_epochs=args.num_epochs,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=10,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.smoke_test:
        args.output_dir = args.output_dir / "smoke_test"  # never clobber a real run
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model_name not in config.SUPPORTED_MODELS:
        print(f"Warning: {args.model_name} is not in config.SUPPORTED_MODELS; "
              "LoRA target modules may need adjusting.")

    model, tokenizer = load_model(args.model_name, quantization=args.quantization)
    train_dataset, eval_dataset = build_datasets(args, tokenizer)

    model = get_peft_model(model, config.get_lora_config(args.model_name))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval=eval_dataset is not None),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Pads input_ids with pad_token_id and labels with -100 (ignored by the loss)
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )

    trainer.train()

    # Save only the LoRA adapter (a few MB) + tokenizer; the base model is reloaded at inference.
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Persist the run config and loss history so results can be reported later.
    run_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with open(args.output_dir / "run_config.json", "w") as f:
        json.dump(run_args, f, indent=2)
    with open(args.output_dir / "log_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    if eval_dataset is not None and not args.smoke_test:
        print("Final eval:", trainer.evaluate())

    print(f"LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
