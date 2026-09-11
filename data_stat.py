import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "master_captions_qc.csv"

CAPTIONS = {
    "Qwen3": "qwen3_caption",
    "Mistral": "mistral_caption",
    "Gemma": "gemma_caption",
}

# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_FILE, sep=",")

print(f"Total samples: {len(df)}")

# ============================================================
# STATISTICS
# ============================================================

results = []

for model, col in CAPTIONS.items():

    captions = df[col].fillna("").astype(str).str.strip()

    # Word count
    word_counts = captions.str.split().str.len()

    # Vocabulary
    words = []
    for caption in captions:
        words.extend(caption.lower().split())

    vocabulary_size = len(set(words))

    # Empty captions
    empty = (captions == "").sum()

    results.append({
        "Model": model,
        "Samples": len(captions),
        "Mean Words": word_counts.mean(),
        "Median Words": word_counts.median(),
        "Std Words": word_counts.std(),
        "Min Words": word_counts.min(),
        "Max Words": word_counts.max(),
        "Vocabulary Size": vocabulary_size,
        "Empty (%)": 100 * empty / len(captions),
    })

# ============================================================
# PRINT
# ============================================================

stats = pd.DataFrame(results)

print("\n" + "=" * 100)
print("CAPTION DATASET STATISTICS")
print("=" * 100)

print(
    stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

# ============================================================
# SAVE
# ============================================================

stats.to_csv(
    "caption_dataset_statistics.csv",
    index=False
)

print("\nSaved to: caption_dataset_statistics.csv")
