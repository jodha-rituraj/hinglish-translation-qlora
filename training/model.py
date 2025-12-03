from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_name):
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit = True,
            device_map = "auto"
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype = torch.float16,
            device_map = "auto"
        )
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer