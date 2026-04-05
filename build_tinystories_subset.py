import re
from datasets import load_dataset

TARGET_CHARS = 400000
OUTPUT_PATH = "data/input.txt"

# Load dataset
ds = load_dataset("roneneldan/TinyStories", split="train")

chunks = []
total_chars = 0

for row in ds:
    text = row["text"]

    # --- FIX ENCODING ISSUES ---
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Replace fancy quotes with normal ones
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("’", "'")

    # Remove non-ASCII garbage characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    if not text:
        continue

    # Add spacing between stories
    block = text + "\n\n"

    chunks.append(block)
    total_chars += len(block)

    if total_chars >= TARGET_CHARS:
        break

final_text = "".join(chunks)[:TARGET_CHARS]

# Save cleaned dataset
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(final_text)

print(f"Saved {len(final_text)} characters to {OUTPUT_PATH}")