import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "master_captions_qc.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"

# Save ImageBind checkpoint somewhere with enough space
MODEL_DIR = "/home/dhanunjaya/scratch/models/imagebind"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "imagebind_huge.pth")

# Output files
OUTPUT_EMBEDDINGS = "imagebind_embeddings.pt"
OUTPUT_SCORES_CSV = "imagebind_similarity_scores.csv"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Process one sample at a time.
# Video preprocessing is memory-heavy, so this is the safest setting.
BATCH_SIZE = 1


# ============================================================
# DOWNLOAD IMAGEBIND CHECKPOINT TO CUSTOM LOCATION
# ============================================================

IMAGEBIND_URL = (
    "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
)


def download_checkpoint():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint already exists:\n{CHECKPOINT_PATH}")
        return

    print("Downloading ImageBind checkpoint...")
    print(f"Saving to: {CHECKPOINT_PATH}")

    urllib.request.urlretrieve(
        IMAGEBIND_URL,
        CHECKPOINT_PATH
    )

    print("Download completed.")


# ============================================================
# LOAD IMAGEBIND
# ============================================================

def load_imagebind_model():

    download_checkpoint()

    print("\nLoading ImageBind model...")

    # IMPORTANT:
    # pretrained=False prevents ImageBind from downloading into
    # its hard-coded .checkpoints directory.
    model = imagebind_model.imagebind_huge(pretrained=False)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(checkpoint)

    model.eval()
    model.to(DEVICE)

    print(f"ImageBind loaded on {DEVICE}")

    return model


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(x, y):
    """
    x, y: [1, D] or [D]

    Returns scalar cosine similarity.
    """

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if y.ndim == 1:
        y = y.unsqueeze(0)

    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)

    return (x * y).sum(dim=-1).item()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ImageBind Audio / Video / Caption Embedding Extraction")
    print("=" * 70)

    print(f"\nInput CSV : {INPUT_CSV}")
    print(f"Data root : {DATA_ROOT}")
    print(f"Device    : {DEVICE}")

    df = pd.read_csv(INPUT_CSV)

    print(f"\nNumber of rows: {len(df)}")

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = [
        "scene_label",
        "filename_audio",
        "filename_video",
        "qwen3_caption",
        "mistral_caption",
        "gemma_caption",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_imagebind_model()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    records = []

    audio_embeddings = []
    video_embeddings = []

    qwen3_embeddings = []
    mistral_embeddings = []
    gemma_embeddings = []

    qwen3_audio_sim = []
    qwen3_video_sim = []

    mistral_audio_sim = []
    mistral_video_sim = []

    gemma_audio_sim = []
    gemma_video_sim = []

    failed_rows = []

    # --------------------------------------------------------
    # Process dataset
    # --------------------------------------------------------

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Extracting ImageBind embeddings"
    ):

        try:

            # =================================================
            # PATHS
            # =================================================

            audio_path = os.path.join(
                DATA_ROOT,
                str(row["filename_audio"])
            )

            video_path = os.path.join(
                DATA_ROOT,
                str(row["filename_video"])
            )

            if not os.path.exists(audio_path):
                raise FileNotFoundError(
                    f"Audio not found: {audio_path}"
                )

            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Video not found: {video_path}"
                )

            # =================================================
            # TEXT
            # =================================================

            qwen3_text = str(
                row["qwen3_caption"]
            ).strip()

            mistral_text = str(
                row["mistral_caption"]
            ).strip()

            gemma_text = str(
                row["gemma_caption"]
            ).strip()

            # Handle NaNs
            if pd.isna(row["qwen3_caption"]):
                qwen3_text = ""

            if pd.isna(row["mistral_caption"]):
                mistral_text = ""

            if pd.isna(row["gemma_caption"]):
                gemma_text = ""

            # =================================================
            # PREPROCESS ALL MODALITIES
            # =================================================

            audio_input = data.load_and_transform_audio_data(
                [audio_path],
                DEVICE
            )

            video_input = data.load_and_transform_video_data(
                [video_path],
                DEVICE
            )

            # Texts can be encoded together
            text_input = data.load_and_transform_text(
                [
                    qwen3_text,
                    mistral_text,
                    gemma_text
                ],
                DEVICE
            )

            # =================================================
            # IMAGEBIND FORWARD PASS
            # =================================================

            inputs = {
                ModalityType.AUDIO: audio_input,
                ModalityType.VISION: video_input,
                ModalityType.TEXT: text_input,
            }

            with torch.no_grad():

                outputs = model(inputs)

            # =================================================
            # EXTRACT EMBEDDINGS
            # =================================================

            # Audio shape:
            # [1, 1024]
            audio_emb = outputs[
                ModalityType.AUDIO
            ][0]

            # Video shape:
            # [1, 1024]
            video_emb = outputs[
                ModalityType.VISION
            ][0]

            # Text shape:
            # [3, 1024]
            text_embs = outputs[
                ModalityType.TEXT
            ]

            qwen3_emb = text_embs[0]
            mistral_emb = text_embs[1]
            gemma_emb = text_embs[2]

            # =================================================
            # COSINE SIMILARITIES
            # =================================================

            q_audio = cosine_similarity(
                audio_emb,
                qwen3_emb
            )

            q_video = cosine_similarity(
                video_emb,
                qwen3_emb
            )

            m_audio = cosine_similarity(
                audio_emb,
                mistral_emb
            )

            m_video = cosine_similarity(
                video_emb,
                mistral_emb
            )

            g_audio = cosine_similarity(
                audio_emb,
                gemma_emb
            )

            g_video = cosine_similarity(
                video_emb,
                gemma_emb
            )

            # =================================================
            # STORE EMBEDDINGS ON CPU
            # =================================================

            audio_embeddings.append(
                audio_emb.detach().cpu()
            )

            video_embeddings.append(
                video_emb.detach().cpu()
            )

            qwen3_embeddings.append(
                qwen3_emb.detach().cpu()
            )

            mistral_embeddings.append(
                mistral_emb.detach().cpu()
            )

            gemma_embeddings.append(
                gemma_emb.detach().cpu()
            )

            # =================================================
            # STORE SIMILARITIES
            # =================================================

            qwen3_audio_sim.append(q_audio)
            qwen3_video_sim.append(q_video)

            mistral_audio_sim.append(m_audio)
            mistral_video_sim.append(m_video)

            gemma_audio_sim.append(g_audio)
            gemma_video_sim.append(g_video)

            records.append({

                "original_index": idx,

                "scene_label":
                    row["scene_label"],

                "filename_audio":
                    row["filename_audio"],

                "filename_video":
                    row["filename_video"],

                # Qwen3
                "qwen3_audio_similarity":
                    q_audio,

                "qwen3_video_similarity":
                    q_video,

                # Mistral
                "mistral_audio_similarity":
                    m_audio,

                "mistral_video_similarity":
                    m_video,

                # Gemma
                "gemma_audio_similarity":
                    g_audio,

                "gemma_video_similarity":
                    g_video,
            })

            # release preprocessing tensors
            del (
                audio_input,
                video_input,
                text_input,
                inputs,
                outputs
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            print(
                f"\nERROR at row {idx}: {e}"
            )

            failed_rows.append({
                "index": idx,
                "error": str(e),
            })

            continue

    # ========================================================
    # STACK EMBEDDINGS
    # ========================================================

    audio_embeddings = torch.stack(
        audio_embeddings
    )

    video_embeddings = torch.stack(
        video_embeddings
    )

    qwen3_embeddings = torch.stack(
        qwen3_embeddings
    )

    mistral_embeddings = torch.stack(
        mistral_embeddings
    )

    gemma_embeddings = torch.stack(
        gemma_embeddings
    )

    # ========================================================
    # SAVE EMBEDDINGS
    # ========================================================

    embedding_data = {

        # Metadata
        "records": records,

        # Shared modalities
        "audio_embeddings":
            audio_embeddings,

        "video_embeddings":
            video_embeddings,

        # Text models
        "qwen3_text_embeddings":
            qwen3_embeddings,

        "mistral_text_embeddings":
            mistral_embeddings,

        "gemma_text_embeddings":
            gemma_embeddings,

        # Similarities
        "qwen3_audio_similarity":
            torch.tensor(qwen3_audio_sim),

        "qwen3_video_similarity":
            torch.tensor(qwen3_video_sim),

        "mistral_audio_similarity":
            torch.tensor(mistral_audio_sim),

        "mistral_video_similarity":
            torch.tensor(mistral_video_sim),

        "gemma_audio_similarity":
            torch.tensor(gemma_audio_sim),

        "gemma_video_similarity":
            torch.tensor(gemma_video_sim),

        # Failed rows
        "failed_rows":
            failed_rows,
    }

    torch.save(
        embedding_data,
        OUTPUT_EMBEDDINGS
    )

    print(
        f"\nEmbeddings saved to: "
        f"{OUTPUT_EMBEDDINGS}"
    )

    # ========================================================
    # SAVE SIMILARITY CSV
    # ========================================================

    score_df = pd.DataFrame(records)

    score_df.to_csv(
        OUTPUT_SCORES_CSV,
        index=False
    )

    print(
        f"Similarity scores saved to: "
        f"{OUTPUT_SCORES_CSV}"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("IMAGEBIND SIMILARITY STATISTICS")
    print("=" * 70)

    models = {

        "QWEN3": {
            "Audio":
                np.array(qwen3_audio_sim),

            "Video":
                np.array(qwen3_video_sim),
        },

        "MISTRAL": {
            "Audio":
                np.array(mistral_audio_sim),

            "Video":
                np.array(mistral_video_sim),
        },

        "GEMMA": {
            "Audio":
                np.array(gemma_audio_sim),

            "Video":
                np.array(gemma_video_sim),
        }
    }

    for model_name, sims in models.items():

        print(f"\n{model_name}")

        for modality, values in sims.items():

            print(f"\n{modality} ↔ Text")

            print(
                f"Mean: {np.mean(values):.6f}"
            )

            print(
                f"P10:  {np.percentile(values, 10):.6f}"
            )

            print(
                f"Std:  {np.std(values):.6f}"
            )

    # ========================================================
    # OPTIONAL COMBINED AUDIO+VIDEO SCORE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("COMBINED AUDIO + VIDEO SIMILARITY")
    print("=" * 70)

    combined = {

        "QWEN3":
            (
                np.array(qwen3_audio_sim)
                +
                np.array(qwen3_video_sim)
            ) / 2,

        "MISTRAL":
            (
                np.array(mistral_audio_sim)
                +
                np.array(mistral_video_sim)
            ) / 2,

        "GEMMA":
            (
                np.array(gemma_audio_sim)
                +
                np.array(gemma_video_sim)
            ) / 2,
    }

    for model_name, values in combined.items():

        print(f"\n{model_name}")

        print(
            f"Mean: {np.mean(values):.6f}"
        )

        print(
            f"P10:  {np.percentile(values, 10):.6f}"
        )

        print(
            f"Std:  {np.std(values):.6f}"
        )

    print("\n")

    if failed_rows:

        print(
            f"Failed rows: {len(failed_rows)}"
        )

        failed_df = pd.DataFrame(
            failed_rows
        )

        failed_df.to_csv(
            "imagebind_failed_rows.csv",
            index=False
        )

        print(
            "Saved failures to "
            "imagebind_failed_rows.csv"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
