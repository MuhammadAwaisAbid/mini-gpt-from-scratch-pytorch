import json
import os
import time
import torch
from torch.optim import AdamW

from utils import (
    load_text,
    build_vocab,
    encode,
    decode,
    save_vocab,
    estimate_loss,
    get_batch,
)
from model import MiniGPT


DATA_PATH = "data/input.txt"
OUTPUT_DIR = "outputs"

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_char_gpt_model.pt")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_char_gpt_model.pt")
VOCAB_PATH = os.path.join(OUTPUT_DIR, "char_vocab.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "char_config.json")

BATCH_SIZE = 16
BLOCK_SIZE = 160
MAX_ITERS = 5000
EVAL_INTERVAL = 100
EVAL_ITERS = 30
LEARNING_RATE = 3e-4

N_EMBD = 96
N_HEAD = 4
N_LAYER = 3
DROPOUT = 0.2

TRAIN_SPLIT = 0.9
SEED = 1337
EARLY_STOPPING_PATIENCE = 6


torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("===== SETUP =====")
print(f"Device: {device}")
print(f"Output directory: {OUTPUT_DIR}")

text = load_text(DATA_PATH)

chars, stoi, itos = build_vocab(text)
vocab_size = len(chars)

encoded_data = encode(text, stoi)
data = torch.tensor(encoded_data, dtype=torch.long)

split_index = int(TRAIN_SPLIT * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

if len(train_data) <= BLOCK_SIZE:
    raise ValueError(
        f"Training split too small for BLOCK_SIZE={BLOCK_SIZE}. Train length={len(train_data)}"
    )

if len(val_data) <= BLOCK_SIZE:
    raise ValueError(
        f"Validation split too small for BLOCK_SIZE={BLOCK_SIZE}. "
        f"Val length={len(val_data)}. Reduce BLOCK_SIZE or add more text."
    )

print("\n===== DATA INFO =====")
print(f"Total characters: {len(text)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Train tokens: {len(train_data)}")
print(f"Val tokens: {len(val_data)}")
print(f"Vocabulary: {chars}")

model = MiniGPT(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    dropout=DROPOUT
).to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

print("\n===== MODEL INFO =====")
print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

config = {
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "dropout": DROPOUT,
    "vocab_size": vocab_size,
    "batch_size": BATCH_SIZE,
    "max_iters": MAX_ITERS,
    "eval_interval": EVAL_INTERVAL,
    "eval_iters": EVAL_ITERS,
    "learning_rate": LEARNING_RATE,
    "train_split": TRAIN_SPLIT,
    "seed": SEED,
    "tokenization": "character_level",
    "top_k_default": 12,
    "temperature_default": 0.75,
    "repetition_penalty_default": 1.03,
    "repetition_window_default": 60,
    "recommended_prompt": "Once upon a time"
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

save_vocab(stoi, itos, VOCAB_PATH)

best_val_loss = float("inf")
best_step = -1
patience_counter = 0
train_start_time = time.time()

print("\n===== TRAINING STARTED =====")

for step in range(MAX_ITERS):
    xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(
            model=model,
            train_data=train_data,
            val_data=val_data,
            eval_iters=EVAL_ITERS,
            block_size=BLOCK_SIZE,
            batch_size=BATCH_SIZE,
            device=device,
        )

        print(
            f"Step {step:4d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            best_step = step
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> Best model saved at step {step}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

total_time = time.time() - train_start_time
torch.save(model.state_dict(), FINAL_MODEL_PATH)

print("\n===== TRAINING COMPLETE =====")
print(f"Best validation loss: {best_val_loss:.4f} at step {best_step}")
print(f"Training time: {total_time:.2f} seconds")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
print(f"Vocab saved to: {VOCAB_PATH}")
print(f"Config saved to: {CONFIG_PATH}")

model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

start_text = "Once upon a time"
context = torch.tensor([[stoi[ch] for ch in start_text]], dtype=torch.long, device=device)

generated_ids = model.generate(
    idx=context,
    max_new_tokens=400,
    temperature=0.75,
    top_k=12,
    repetition_penalty=1.03,
    repetition_window=60
)[0].tolist()

generated_text = decode(generated_ids, itos)

print("\n===== SAMPLE GENERATED TEXT (BEST MODEL) =====")
print(generated_text)