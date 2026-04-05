import json
import os
import torch


def load_text(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        raise ValueError("Input text file is empty. Add text to data/input.txt")

    return text


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(text: str, stoi: dict) -> list[int]:
    unknown_chars = [ch for ch in text if ch not in stoi]
    if unknown_chars:
        raise ValueError(f"Unknown characters found during encoding: {set(unknown_chars)}")
    return [stoi[ch] for ch in text]


def decode(indices: list[int], itos: dict) -> str:
    return "".join([itos[i] for i in indices])


def save_vocab(stoi: dict, itos: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stoi": stoi,
                "itos": {str(k): v for k, v in itos.items()}
            },
            f,
            ensure_ascii=False,
            indent=2
        )


def load_vocab(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vocab file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stoi = data["stoi"]
    itos = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device
):
    if len(data) <= block_size:
        raise ValueError(
            f"Dataset split is too small for block_size={block_size}. "
            f"Split length={len(data)}"
        )

    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    eval_iters: int,
    block_size: int,
    batch_size: int,
    device: torch.device
):
    model.eval()
    out = {}

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)

        for i in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()

        out[split_name] = losses.mean().item()

    model.train()
    return out