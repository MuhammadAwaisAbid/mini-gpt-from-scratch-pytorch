import json
import os
import torch

from utils import load_vocab, decode
from model import MiniGPT


OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_char_gpt_model.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "char_vocab.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "char_config.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. Train the model first."
        )

    stoi, itos = load_vocab(VOCAB_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    model = MiniGPT(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("===== CHARACTER-LEVEL MINI GPT TEXT GENERATION =====")
    prompt = input("Enter a prompt (example: Once upon a time): ")

    if not prompt:
        prompt = "Once upon a time"

    for ch in prompt:
        if ch not in stoi:
            raise ValueError(
                f"Character {repr(ch)} not found in vocabulary. "
                f"Use only characters that exist in data/input.txt"
            )

    max_new_tokens_input = input("Enter max new tokens (default 400): ").strip()
    temperature_input = input("Enter temperature (default 0.65): ").strip()
    top_k_input = input("Enter top-k (default 8): ").strip()
    repetition_penalty_input = input("Enter repetition penalty (default 1.15): ").strip()

    max_new_tokens = int(max_new_tokens_input) if max_new_tokens_input else 400
    temperature = float(temperature_input) if temperature_input else 0.65
    top_k = int(top_k_input) if top_k_input else 8
    repetition_penalty = float(repetition_penalty_input) if repetition_penalty_input else 1.15

    prompt_ids = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long, device=device)

    generated_ids = model.generate(
        idx=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=60
    )[0].tolist()

    generated_text = decode(generated_ids, itos)

    print("\n===== GENERATED TEXT =====")
    print(generated_text)


if __name__ == "__main__":
    main()