import os

# ============================================================
# GPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import librosa
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
import laion_clap


# ============================================================
# PATHS
# ============================================================
INPUT_CSV = "master_captions_qc.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"

OUTPUT_FILE = "qwen3_clap_embeddings_and_scores.pt"


# ============================================================
# AUDIO SETTINGS
# ============================================================
TARGET_SR = 48000


# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_CSV)

print("Total rows:", len(df))


# ============================================================
# CAPTION COLUMN
# ============================================================
CAPTION_COLUMN = "qwen3_caption"


# ============================================================
# VALID SAMPLES
# ============================================================
def valid_caption(x):

    if pd.isna(x):
        return False

    return len(str(x).strip()) > 0


valid = df[CAPTION_COLUMN].apply(
    valid_caption
).values


valid &= df["filename_audio"].apply(
    lambda x: os.path.isfile(
        os.path.join(
            DATA_ROOT,
            str(x),
        )
    )
).values


eval_df = df.loc[valid].copy()

print(
    "Valid samples:",
    len(eval_df),
)


# ============================================================
# LOAD CLAP
# ============================================================
print("\nLoading CLAP...")

clap_model = laion_clap.CLAP_Module(
    enable_fusion=False
)

clap_model.load_ckpt()

print("CLAP loaded.")


# ============================================================
# STORAGE
# ============================================================
audio_embeddings = []
text_embeddings = []

clap_similarities = []

filenames = []
scene_labels = []
captions = []


# ============================================================
# EXTRACTION
# ============================================================
print("\nExtracting CLAP embeddings and similarities...")


for idx, row in tqdm(
    eval_df.iterrows(),
    total=len(eval_df),
    desc="CLAP",
):

    audio_path = os.path.join(
        DATA_ROOT,
        str(row["filename_audio"]),
    )

    caption = str(
        row[CAPTION_COLUMN]
    ).strip()


    # ========================================================
    # LOAD AUDIO
    # ========================================================
    try:

        audio, sr = librosa.load(
            audio_path,
            sr=TARGET_SR,
            mono=True,
        )

        audio = audio.astype(
            np.float32
        )

    except Exception as e:

        print(
            f"\nAudio error "
            f"row={idx}: "
            f"{type(e).__name__}: {e}"
        )

        continue


    # ========================================================
    # AUDIO EMBEDDING
    # ========================================================
    try:

        audio_input = audio.reshape(
            1,
            -1,
        )

        audio_embedding = (
            clap_model
            .get_audio_embedding_from_data(
                x=audio_input,
                use_tensor=False,
            )
        )

    except Exception as e:

        print(
            f"\nAudio embedding error "
            f"row={idx}: "
            f"{type(e).__name__}: {e}"
        )

        continue


    # ========================================================
    # TEXT EMBEDDING
    # ========================================================
    try:

        text_embedding = (
            clap_model
            .get_text_embedding(
                [caption],
                use_tensor=False,
            )
        )

    except Exception as e:

        print(
            f"\nText embedding error "
            f"row={idx}: "
            f"{type(e).__name__}: {e}"
        )

        continue


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================
    audio_embedding = np.asarray(
        audio_embedding,
        dtype=np.float32,
    )[0]

    text_embedding = np.asarray(
        text_embedding,
        dtype=np.float32,
    )[0]


    # ========================================================
    # NORMALIZE
    # ========================================================
    audio_norm = np.linalg.norm(
        audio_embedding
    )

    text_norm = np.linalg.norm(
        text_embedding
    )


    audio_embedding_normalized = (
        audio_embedding /
        max(audio_norm, 1e-12)
    )

    text_embedding_normalized = (
        text_embedding /
        max(text_norm, 1e-12)
    )


    # ========================================================
    # CLAP SIMILARITY
    # ========================================================
    similarity = float(
        np.dot(
            audio_embedding_normalized,
            text_embedding_normalized,
        )
    )


    # ========================================================
    # STORE
    # ========================================================
    audio_embeddings.append(
        audio_embedding
    )

    text_embeddings.append(
        text_embedding
    )

    clap_similarities.append(
        similarity
    )

    filenames.append(
        row["filename_audio"]
    )

    scene_labels.append(
        row["scene_label"]
    )

    captions.append(
        caption
    )


    # ========================================================
    # CHECKPOINT
    # ========================================================
    if len(audio_embeddings) % 5000 == 0:

        checkpoint = {

            "audio_embeddings":
                torch.from_numpy(
                    np.stack(
                        audio_embeddings
                    )
                ),

            "text_embeddings":
                torch.from_numpy(
                    np.stack(
                        text_embeddings
                    )
                ),

            "clap_similarity":
                torch.tensor(
                    clap_similarities,
                    dtype=torch.float32,
                ),

            "filename_audio":
                filenames,

            "scene_label":
                scene_labels,

            "caption":
                captions,
        }

        torch.save(
            checkpoint,
            OUTPUT_FILE,
        )

        print(
            f"\nCheckpoint saved: "
            f"{len(audio_embeddings)} samples"
        )


# ============================================================
# FINAL ARRAYS
# ============================================================
audio_embeddings = np.stack(
    audio_embeddings
).astype(
    np.float32
)

text_embeddings = np.stack(
    text_embeddings
).astype(
    np.float32
)

clap_similarities = np.asarray(
    clap_similarities,
    dtype=np.float32,
)


# ============================================================
# FINAL SAVE
# ============================================================
data = {

    "audio_embeddings":
        torch.from_numpy(
            audio_embeddings
        ),

    "text_embeddings":
        torch.from_numpy(
            text_embeddings
        ),

    "clap_similarity":
        torch.from_numpy(
            clap_similarities
        ),

    "filename_audio":
        filenames,

    "scene_label":
        scene_labels,

    "caption":
        captions,
}


torch.save(
    data,
    OUTPUT_FILE,
)


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("CLAP EXTRACTION COMPLETE")
print("=" * 70)

print(
    "Audio embeddings shape:",
    data["audio_embeddings"].shape,
)

print(
    "Text embeddings shape:",
    data["text_embeddings"].shape,
)

print(
    "Similarity shape:",
    data["clap_similarity"].shape,
)

print(
    "Number of samples:",
    len(filenames),
)

print(
    "Mean CLAP similarity:",
    data["clap_similarity"].mean().item(),
)

print(
    "Std CLAP similarity:",
    data["clap_similarity"].std().item(),
)

print(
    "Min CLAP similarity:",
    data["clap_similarity"].min().item(),
)

print(
    "Max CLAP similarity:",
    data["clap_similarity"].max().item(),
)

print(
    "\nSaved to:",
    OUTPUT_FILE,
)
