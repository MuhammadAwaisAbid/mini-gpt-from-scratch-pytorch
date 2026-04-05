# Mini GPT from Scratch using PyTorch

A mini GPT-style language model built from scratch in PyTorch to learn transformer fundamentals, autoregressive text generation, and the effect of data scaling on language modeling quality.

## Project Overview

This project implements a small GPT-like decoder-only transformer from scratch using PyTorch.  
The goal is educational: to understand how token embeddings, positional embeddings, masked self-attention, feed-forward layers, residual connections, and autoregressive generation work in practice.

The repository includes:

- a transformer model implemented from scratch
- character-level language modeling
- training and validation pipeline
- text generation script
- dataset preparation script using TinyStories subsets
- experiments showing how dataset size improves output quality

## Features

- Decoder-only transformer architecture
- Multi-head masked self-attention
- Feed-forward transformer blocks
- Character-level tokenization
- Training / validation split
- Best-checkpoint saving
- Config and vocabulary saving
- Interactive text generation
- Repetition penalty and top-k sampling
- TinyStories subset generation for better training data

## Repository Structure

```text
mini-gpt-from-scratch-pytorch/
│
├── data/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── src/
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   └── generate.py
├── build_tinystories_subset.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## Results
The model was trained on progressively larger datasets.

| Dataset Size | Validation Loss | Output Quality |
|-------------|---------------|---------------|
| 50k chars   | ~4.5          | Poor |
| 120k chars  | ~1.7          | Readable |
| 200k chars  | ~1.36         | Good |
| 400k chars  | **~1.25**     | **Strong** |

The final model produces structured short stories with correct punctuation and reasonable grammar.

## Future Work
- Implement subword tokenization (BPE)
- Train larger models with longer context
- Add evaluation metrics and visualizations
- Experiment with different decoding strategies
