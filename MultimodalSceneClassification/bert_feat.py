import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "master_captions_qc.csv"

OUTPUT_FILE = "bert_cls_caption_embeddings.pt"

MODEL_NAME = "bert-base-uncased"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
MAX_LENGTH = 80


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Number of samples:", len(df))

CAPTION_COLUMNS = [
    "qwen3_caption",
    "mistral_caption",
    "gemma_caption",
]

for col in CAPTION_COLUMNS:
    if col not in df.columns:
        raise ValueError(
            f"Missing column: {col}"
        )


# ============================================================
# LOAD BERT
# ============================================================

print("\nLoading BERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("Model       :", MODEL_NAME)
print("Device      :", DEVICE)
print("Embedding   :", model.config.hidden_size)


# ============================================================
# EXTRACT CLS EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_cls_embeddings(
    captions,
    batch_size=64
):

    embeddings = []

    for start in tqdm(
        range(
            0,
            len(captions),
            batch_size
        ),
        desc="Extracting"
    ):

        batch = captions[
            start:start + batch_size
        ]

        # Handle NaN / missing captions
        batch = [
            "" if pd.isna(x)
            else str(x)
            for x in batch
        ]

        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        outputs = model(**encoded)

        # ----------------------------------------------------
        # CLS TOKEN
        # ----------------------------------------------------
        #
        # outputs.last_hidden_state:
        #
        # [batch_size, sequence_length, 768]
        #
        # [:, 0, :] selects the [CLS] token.
        #
        # ----------------------------------------------------

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        # L2 normalization
        cls_embedding = torch.nn.functional.normalize(
            cls_embedding,
            p=2,
            dim=1
        )

        embeddings.append(
            cls_embedding.cpu()
        )

    return torch.cat(
        embeddings,
        dim=0
    )


# ============================================================
# QWEN3
# ============================================================

print("\n" + "=" * 70)
print("Qwen3")
print("=" * 70)

qwen3_embeddings = extract_cls_embeddings(
    df["qwen3_caption"].tolist(),
    BATCH_SIZE
)

print(
    "Qwen3 embeddings:",
    qwen3_embeddings.shape
)


# ============================================================
# MISTRAL
# ============================================================

print("\n" + "=" * 70)
print("Mistral")
print("=" * 70)

mistral_embeddings = extract_cls_embeddings(
    df["mistral_caption"].tolist(),
    BATCH_SIZE
)

print(
    "Mistral embeddings:",
    mistral_embeddings.shape
)


# ============================================================
# GEMMA
# ============================================================

print("\n" + "=" * 70)
print("Gemma")
print("=" * 70)

gemma_embeddings = extract_cls_embeddings(
    df["gemma_caption"].tolist(),
    BATCH_SIZE
)

print(
    "Gemma embeddings:",
    gemma_embeddings.shape
)


# ============================================================
# SAVE
# ============================================================

save_data = {

    "qwen3_embeddings":
        qwen3_embeddings,

    "mistral_embeddings":
        mistral_embeddings,

    "gemma_embeddings":
        gemma_embeddings,

    # Metadata
    "scene_labels":
        df["scene_label"].tolist(),

    "filename_audio":
        df["filename_audio"].tolist(),

    "filename_video":
        df["filename_video"].tolist(),

    "model_name":
        MODEL_NAME,

    "embedding_type":
        "CLS",

    "embedding_dimension":
        qwen3_embeddings.shape[1],
}


torch.save(
    save_data,
    OUTPUT_FILE
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    "Qwen3  :",
    qwen3_embeddings.shape
)

print(
    "Mistral:",
    mistral_embeddings.shape
)

print(
    "Gemma  :",
    gemma_embeddings.shape
)

print(
    "Saved to:",
    OUTPUT_FILE
)
