import os
import torch
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

EMBEDDING_FILE = "bert_cls_caption_embeddings.pt"

TRAIN_CSV = "TAU-urban-audio-visual-scenes/create_data/evaluation_setup/fold1_train.csv"
EVAL_CSV = "TAU-urban-audio-visual-scenes/create_data/evaluation_setup/fold1_evaluate.csv"

OUTPUT_DIR = "bert_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD CAPTION EMBEDDINGS
# ============================================================

data = torch.load(
    EMBEDDING_FILE,
    map_location="cpu"
)

qwen = data["qwen3_embeddings"].cpu().numpy()
mistral = data["mistral_embeddings"].cpu().numpy()
gemma = data["gemma_embeddings"].cpu().numpy()

all_names = np.asarray(data["filename_audio"])

print("Total samples:", len(all_names))
print("Qwen:", qwen.shape)
print("Mistral:", mistral.shape)
print("Gemma:", gemma.shape)


# ============================================================
# LOAD TRAIN / EVAL SPLITS
# ============================================================

train_df = pd.read_csv(TRAIN_CSV, sep="\t")
eval_df = pd.read_csv(EVAL_CSV, sep="\t")

train_names = train_df["filename_audio"].values
eval_names = eval_df["filename_audio"].values


# ============================================================
# FIND INDICES IN ORIGINAL .PT FILE
# ============================================================

name_to_idx = {
    name: i for i, name in enumerate(all_names)
}

train_idx = [name_to_idx[name] for name in train_names]
eval_idx = [name_to_idx[name] for name in eval_names]


# ============================================================
# CHECK
# ============================================================

print("\nTrain samples:", len(train_idx))
print("Eval samples :", len(eval_idx))

assert len(train_idx) == len(train_df)
assert len(eval_idx) == len(eval_df)


# ============================================================
# EXTRACT TRAIN / EVAL EMBEDDINGS
# ============================================================

embeddings = {
    "qwen": qwen,
    "mistral": mistral,
    "gemma": gemma
}

for name, emb in embeddings.items():

    train_emb = emb[train_idx]
    eval_emb = emb[eval_idx]

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_train.npy"
        ),
        train_emb
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_eval.npy"
        ),
        eval_emb
    )

    print(
        f"{name:8s} | "
        f"train: {train_emb.shape} | "
        f"eval: {eval_emb.shape}"
    )


# ============================================================
# SAVE LABELS AND FILENAMES
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "train_labels.npy"),
    train_df["scene_label"].values
)

np.save(
    os.path.join(OUTPUT_DIR, "eval_labels.npy"),
    eval_df["scene_label"].values
)

np.save(
    os.path.join(OUTPUT_DIR, "train_names.npy"),
    train_names
)

np.save(
    os.path.join(OUTPUT_DIR, "eval_names.npy"),
    eval_names
)


print("\nSaved to:", OUTPUT_DIR)
