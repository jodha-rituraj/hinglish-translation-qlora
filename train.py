import argparse
from config import *
from training.dataset import load_training_data, prepare_training_data
from training.model import load_model

def parse_args():
    parser = argparse.ArgumentParser(description = "Training Script for Hinglish Translation using QLoRA")

    parser.add_argument("--model_name", type = str, default = MODEL_NAME, help = "Base model name or path for fine-tuning")
    parser.add_argument("--data_path", type = str, default = DATA_DIR / "hinge_train.json", help="Path to the training data JSON file")
    parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Directory to save outputs and checkpoints")
    parser.add_argument("--batch_size", type = int, default = BATCH_SIZE, help = "Batch size for training")
    parser.add_argument("--num_epochs", type = int, default = NUM_EPOCHS, help = "Number of training epochs")
    parser.add_argument("--learning_rate", type = float, default = LEARNING_RATE, help = "Learning rate for optimizer")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load and prepare data
    raw_data = load_training_data(args.data_path, max_samples = MAX_SAMPLES)
    training_data = prepare_training_data(raw_data)

    print(f"Loaded {len(training_data)} training samples.")
    print(training_data[0]) 

    # Load model and tokenizer
    model, tokenizer = load_model(args.model_name)

    print("Model and tokenizer loaded successfully.")
