import os
import urllib.request

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

# ------------------------------------------------------------
# ImageBind checkpoint
# ------------------------------------------------------------

MODEL_DIR = "/home/dhanunjaya/scratch/models/imagebind"

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "imagebind_huge.pth"
)

# ------------------------------------------------------------
# Output files
# ------------------------------------------------------------

OUTPUT_EMBEDDINGS = (
    "imagebind_embeddings_new.pt"
)

OUTPUT_SCORES_CSV = (
    "imagebind_similarity_scores_new.csv"
)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Process one sample at a time
# ------------------------------------------------------------

BATCH_SIZE = 1


# ============================================================
# DOWNLOAD IMAGEBIND CHECKPOINT
# ============================================================

IMAGEBIND_URL = (
    "https://dl.fbaipublicfiles.com/imagebind/"
    "imagebind_huge.pth"
)


def download_checkpoint():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    if os.path.exists(
        CHECKPOINT_PATH
    ):

        print(
            f"Checkpoint already exists:\n"
            f"{CHECKPOINT_PATH}"
        )

        return

    print(
        "Downloading ImageBind checkpoint..."
    )

    print(
        f"Saving to: {CHECKPOINT_PATH}"
    )

    urllib.request.urlretrieve(
        IMAGEBIND_URL,
        CHECKPOINT_PATH
    )

    print(
        "Download completed."
    )


# ============================================================
# LOAD IMAGEBIND
# ============================================================

def load_imagebind_model():

    download_checkpoint()

    print(
        "\nLoading ImageBind model..."
    )

    # IMPORTANT:
    # pretrained=False prevents ImageBind from downloading
    # to its default .checkpoints directory.

    model = (
        imagebind_model.imagebind_huge(
            pretrained=False
        )
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    model.to(
        DEVICE
    )

    print(
        f"ImageBind loaded on {DEVICE}"
    )

    return model


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    x,
    y
):
    """
    x, y:
        [D] or [1, D]

    Returns:
        scalar cosine similarity
    """

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if y.ndim == 1:
        y = y.unsqueeze(0)

    x = F.normalize(
        x.float(),
        dim=-1
    )

    y = F.normalize(
        y.float(),
        dim=-1
    )

    return (
        (x * y)
        .sum(dim=-1)
        .item()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 80
    )

    print(
        "ImageBind Audio / Video / Text "
        "Embedding Extraction"
    )

    print(
        "=" * 80
    )

    print(
        f"\nInput CSV : {INPUT_CSV}"
    )

    print(
        f"Data root : {DATA_ROOT}"
    )

    print(
        f"Device    : {DEVICE}"
    )

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_CSV
    )

    print(
        f"\nNumber of rows: {len(df)}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [

        "scene_label",

        "filename_audio",

        "filename_video",

        # Stage 1 captions
        "final_audio_caption",
        "final_video_caption",

        # Generated captions
        "qwen3_caption",
        "mistral_caption",
        "gemma_caption",
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in CSV: {missing}"
        )

    # --------------------------------------------------------
    # Load ImageBind
    # --------------------------------------------------------

    model = load_imagebind_model()

    # ========================================================
    # STORAGE
    # ========================================================

    records = []

    # --------------------------------------------------------
    # Modality embeddings
    # --------------------------------------------------------

    audio_embeddings = []

    video_embeddings = []

    # --------------------------------------------------------
    # Text embeddings
    # --------------------------------------------------------

    final_audio_caption_embeddings = []

    final_video_caption_embeddings = []

    qwen3_embeddings = []

    mistral_embeddings = []

    gemma_embeddings = []

    # --------------------------------------------------------
    # Audio similarities
    # --------------------------------------------------------

    final_audio_caption_audio_sim = []

    final_video_caption_audio_sim = []

    qwen3_audio_sim = []

    mistral_audio_sim = []

    gemma_audio_sim = []

    # --------------------------------------------------------
    # Video similarities
    # --------------------------------------------------------

    final_audio_caption_video_sim = []

    final_video_caption_video_sim = []

    qwen3_video_sim = []

    mistral_video_sim = []

    gemma_video_sim = []

    # --------------------------------------------------------
    # Failed rows
    # --------------------------------------------------------

    failed_rows = []

    # ========================================================
    # PROCESS DATASET
    # ========================================================

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

            if not os.path.exists(
                audio_path
            ):

                raise FileNotFoundError(
                    f"Audio not found: {audio_path}"
                )

            if not os.path.exists(
                video_path
            ):

                raise FileNotFoundError(
                    f"Video not found: {video_path}"
                )

            # =================================================
            # TEXT
            # =================================================

            def clean_caption(value):

                if pd.isna(value):
                    return ""

                return str(value).strip()


            final_audio_text = clean_caption(
                row["final_audio_caption"]
            )

            final_video_text = clean_caption(
                row["final_video_caption"]
            )

            qwen3_text = clean_caption(
                row["qwen3_caption"]
            )

            mistral_text = clean_caption(
                row["mistral_caption"]
            )

            gemma_text = clean_caption(
                row["gemma_caption"]
            )

            # =================================================
            # PREPROCESS AUDIO
            # =================================================

            audio_input = (
                data.load_and_transform_audio_data(
                    [audio_path],
                    DEVICE
                )
            )

            # =================================================
            # PREPROCESS VIDEO
            # =================================================

            video_input = (
                data.load_and_transform_video_data(
                    [video_path],
                    DEVICE
                )
            )

            # =================================================
            # PREPROCESS TEXT
            # =================================================

            text_input = (
                data.load_and_transform_text(
                    [
                        final_audio_text,
                        final_video_text,
                        qwen3_text,
                        mistral_text,
                        gemma_text
                    ],
                    DEVICE
                )
            )

            # =================================================
            # IMAGEBIND FORWARD PASS
            # =================================================

            inputs = {

                ModalityType.AUDIO:
                    audio_input,

                ModalityType.VISION:
                    video_input,

                ModalityType.TEXT:
                    text_input,
            }

            with torch.no_grad():

                outputs = model(
                    inputs
                )

            # =================================================
            # AUDIO EMBEDDING
            # =================================================

            audio_emb = (
                outputs[
                    ModalityType.AUDIO
                ][0]
            )

            # =================================================
            # VIDEO EMBEDDING
            # =================================================

            video_emb = (
                outputs[
                    ModalityType.VISION
                ][0]
            )

            # =================================================
            # TEXT EMBEDDINGS
            #
            # Order:
            #
            # 0 -> final_audio_caption
            # 1 -> final_video_caption
            # 2 -> qwen3
            # 3 -> mistral
            # 4 -> gemma
            # =================================================

            text_embs = (
                outputs[
                    ModalityType.TEXT
                ]
            )

            final_audio_emb = (
                text_embs[0]
            )

            final_video_emb = (
                text_embs[1]
            )

            qwen3_emb = (
                text_embs[2]
            )

            mistral_emb = (
                text_embs[3]
            )

            gemma_emb = (
                text_embs[4]
            )

            # =================================================
            # AUDIO ↔ TEXT
            # =================================================

            fa_audio = cosine_similarity(
                audio_emb,
                final_audio_emb
            )

            fv_audio = cosine_similarity(
                audio_emb,
                final_video_emb
            )

            q_audio = cosine_similarity(
                audio_emb,
                qwen3_emb
            )

            m_audio = cosine_similarity(
                audio_emb,
                mistral_emb
            )

            g_audio = cosine_similarity(
                audio_emb,
                gemma_emb
            )

            # =================================================
            # VIDEO ↔ TEXT
            # =================================================

            fa_video = cosine_similarity(
                video_emb,
                final_audio_emb
            )

            fv_video = cosine_similarity(
                video_emb,
                final_video_emb
            )

            q_video = cosine_similarity(
                video_emb,
                qwen3_emb
            )

            m_video = cosine_similarity(
                video_emb,
                mistral_emb
            )

            g_video = cosine_similarity(
                video_emb,
                gemma_emb
            )

            # =================================================
            # STORE AUDIO EMBEDDING
            # =================================================

            audio_embeddings.append(
                audio_emb
                .detach()
                .cpu()
                .float()
            )

            # =================================================
            # STORE VIDEO EMBEDDING
            # =================================================

            video_embeddings.append(
                video_emb
                .detach()
                .cpu()
                .float()
            )

            # =================================================
            # STORE TEXT EMBEDDINGS
            # =================================================

            final_audio_caption_embeddings.append(
                final_audio_emb
                .detach()
                .cpu()
                .float()
            )

            final_video_caption_embeddings.append(
                final_video_emb
                .detach()
                .cpu()
                .float()
            )

            qwen3_embeddings.append(
                qwen3_emb
                .detach()
                .cpu()
                .float()
            )

            mistral_embeddings.append(
                mistral_emb
                .detach()
                .cpu()
                .float()
            )

            gemma_embeddings.append(
                gemma_emb
                .detach()
                .cpu()
                .float()
            )

            # =================================================
            # STORE AUDIO SIMILARITIES
            # =================================================

            final_audio_caption_audio_sim.append(
                fa_audio
            )

            final_video_caption_audio_sim.append(
                fv_audio
            )

            qwen3_audio_sim.append(
                q_audio
            )

            mistral_audio_sim.append(
                m_audio
            )

            gemma_audio_sim.append(
                g_audio
            )

            # =================================================
            # STORE VIDEO SIMILARITIES
            # =================================================

            final_audio_caption_video_sim.append(
                fa_video
            )

            final_video_caption_video_sim.append(
                fv_video
            )

            qwen3_video_sim.append(
                q_video
            )

            mistral_video_sim.append(
                m_video
            )

            gemma_video_sim.append(
                g_video
            )

            # =================================================
            # RECORD
            # =================================================

            records.append({

                "original_index":
                    idx,

                "scene_label":
                    row["scene_label"],

                "filename_audio":
                    row["filename_audio"],

                "filename_video":
                    row["filename_video"],

                # ------------------------------------------------
                # Audio ↔ Caption
                # ------------------------------------------------

                "final_audio_caption_audio_similarity":
                    fa_audio,

                "final_video_caption_audio_similarity":
                    fv_audio,

                "qwen3_audio_similarity":
                    q_audio,

                "mistral_audio_similarity":
                    m_audio,

                "gemma_audio_similarity":
                    g_audio,

                # ------------------------------------------------
                # Video ↔ Caption
                # ------------------------------------------------

                "final_audio_caption_video_similarity":
                    fa_video,

                "final_video_caption_video_similarity":
                    fv_video,

                "qwen3_video_similarity":
                    q_video,

                "mistral_video_similarity":
                    m_video,

                "gemma_video_similarity":
                    g_video,
            })

            # =================================================
            # RELEASE GPU MEMORY
            # =================================================

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

                "index":
                    idx,

                "error":
                    str(e)
            })

            continue

    # ========================================================
    # CHECK
    # ========================================================

    if len(audio_embeddings) == 0:

        raise RuntimeError(
            "No samples were successfully processed."
        )

    # ========================================================
    # STACK EMBEDDINGS
    # ========================================================

    audio_embeddings = torch.stack(
        audio_embeddings
    )

    video_embeddings = torch.stack(
        video_embeddings
    )

    final_audio_caption_embeddings = (
        torch.stack(
            final_audio_caption_embeddings
        )
    )

    final_video_caption_embeddings = (
        torch.stack(
            final_video_caption_embeddings
        )
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
    # CONVERT SIMILARITIES TO TENSORS
    # ========================================================

    final_audio_caption_audio_sim = torch.tensor(
        final_audio_caption_audio_sim,
        dtype=torch.float32
    )

    final_video_caption_audio_sim = torch.tensor(
        final_video_caption_audio_sim,
        dtype=torch.float32
    )

    qwen3_audio_sim = torch.tensor(
        qwen3_audio_sim,
        dtype=torch.float32
    )

    mistral_audio_sim = torch.tensor(
        mistral_audio_sim,
        dtype=torch.float32
    )

    gemma_audio_sim = torch.tensor(
        gemma_audio_sim,
        dtype=torch.float32
    )

    final_audio_caption_video_sim = torch.tensor(
        final_audio_caption_video_sim,
        dtype=torch.float32
    )

    final_video_caption_video_sim = torch.tensor(
        final_video_caption_video_sim,
        dtype=torch.float32
    )

    qwen3_video_sim = torch.tensor(
        qwen3_video_sim,
        dtype=torch.float32
    )

    mistral_video_sim = torch.tensor(
        mistral_video_sim,
        dtype=torch.float32
    )

    gemma_video_sim = torch.tensor(
        gemma_video_sim,
        dtype=torch.float32
    )

    # ========================================================
    # SAVE EMBEDDINGS
    # ========================================================

    embedding_data = {

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        "records":
            records,

        "scene_labels":
            [
                x["scene_label"]
                for x in records
            ],

        "filename_audio":
            [
                x["filename_audio"]
                for x in records
            ],

        "filename_video":
            [
                x["filename_video"]
                for x in records
            ],

        # ----------------------------------------------------
        # Audio embedding
        # ----------------------------------------------------

        "audio_embeddings":
            audio_embeddings,

        # ----------------------------------------------------
        # Video embedding
        # ----------------------------------------------------

        "video_embeddings":
            video_embeddings,

        # ----------------------------------------------------
        # Text embeddings
        # ----------------------------------------------------

        "final_audio_caption_embeddings":
            final_audio_caption_embeddings,

        "final_video_caption_embeddings":
            final_video_caption_embeddings,

        "qwen3_text_embeddings":
            qwen3_embeddings,

        "mistral_text_embeddings":
            mistral_embeddings,

        "gemma_text_embeddings":
            gemma_embeddings,

        # ----------------------------------------------------
        # Audio ↔ Text similarities
        # ----------------------------------------------------

        "audio_final_audio_caption_similarity":
            final_audio_caption_audio_sim,

        "audio_final_video_caption_similarity":
            final_video_caption_audio_sim,

        "audio_qwen3_similarity":
            qwen3_audio_sim,

        "audio_mistral_similarity":
            mistral_audio_sim,

        "audio_gemma_similarity":
            gemma_audio_sim,

        # ----------------------------------------------------
        # Video ↔ Text similarities
        # ----------------------------------------------------

        "video_final_audio_caption_similarity":
            final_audio_caption_video_sim,

        "video_final_video_caption_similarity":
            final_video_caption_video_sim,

        "video_qwen3_similarity":
            qwen3_video_sim,

        "video_mistral_similarity":
            mistral_video_sim,

        "video_gemma_similarity":
            gemma_video_sim,

        # ----------------------------------------------------
        # Failed rows
        # ----------------------------------------------------

        "failed_rows":
            failed_rows,

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        "model":
            "ImageBind-Huge",

        "checkpoint_path":
            CHECKPOINT_PATH,

        "embedding_dimension":
            audio_embeddings.shape[1],

        "embedding_normalized":
            False,
    }

    torch.save(
        embedding_data,
        OUTPUT_EMBEDDINGS
    )

    print(
        f"\nEmbeddings saved to:\n"
        f"{OUTPUT_EMBEDDINGS}"
    )

    # ========================================================
    # SAVE SIMILARITY CSV
    # ========================================================

    score_df = pd.DataFrame(
        records
    )

    score_df.to_csv(
        OUTPUT_SCORES_CSV,
        index=False
    )

    print(
        f"Similarity scores saved to:\n"
        f"{OUTPUT_SCORES_CSV}"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        "IMAGEBIND AUDIO ↔ TEXT SIMILARITY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Model':<25}"
        f"{'Mean':>12}"
        f"{'P10':>12}"
        f"{'Std':>12}"
    )

    print(
        "-" * 65
    )

    audio_results = {

        "STAGE1_AUDIO":
            final_audio_caption_audio_sim,

        "STAGE1_VIDEO":
            final_video_caption_audio_sim,

        "QWEN3":
            qwen3_audio_sim,

        "MISTRAL":
            mistral_audio_sim,

        "GEMMA":
            gemma_audio_sim,
    }

    for name, values in audio_results.items():

        values = values.float()

        mean = values.mean().item()

        p10 = torch.quantile(
            values,
            0.10
        ).item()

        std = values.std().item()

        print(
            f"{name:<25}"
            f"{mean:>12.6f}"
            f"{p10:>12.6f}"
            f"{std:>12.6f}"
        )

    # ========================================================
    # VIDEO RESULTS
    # ========================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        "IMAGEBIND VIDEO ↔ TEXT SIMILARITY"
    )

    print(
        "=" * 100
    )

    print(
        f"{'Model':<25}"
        f"{'Mean':>12}"
        f"{'P10':>12}"
        f"{'Std':>12}"
    )

    print(
        "-" * 65
    )

    video_results = {

        "STAGE1_AUDIO":
            final_audio_caption_video_sim,

        "STAGE1_VIDEO":
            final_video_caption_video_sim,

        "QWEN3":
            qwen3_video_sim,

        "MISTRAL":
            mistral_video_sim,

        "GEMMA":
            gemma_video_sim,
    }

    for name, values in video_results.items():

        values = values.float()

        mean = values.mean().item()

        p10 = torch.quantile(
            values,
            0.10
        ).item()

        std = values.std().item()

        print(
            f"{name:<25}"
            f"{mean:>12.6f}"
            f"{p10:>12.6f}"
            f"{std:>12.6f}"
        )

    # ========================================================
    # FAILED ROWS
    # ========================================================

    print("\n")

    print(
        f"Successful samples: "
        f"{len(audio_embeddings)}"
    )

    print(
        f"Failed samples: "
        f"{len(failed_rows)}"
    )

    if failed_rows:

        failed_df = pd.DataFrame(
            failed_rows
        )

        failed_df.to_csv(
            "imagebind_failed_rows.csv",
            index=False
        )

        print(
            "Failed rows saved to:"
        )

        print(
            "imagebind_failed_rows.csv"
        )

    # ========================================================
    # FINAL SHAPES
    # ========================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        "FINAL EMBEDDING SHAPES"
    )

    print(
        "=" * 100
    )

    print(
        "Audio                  :",
        audio_embeddings.shape
    )

    print(
        "Video                  :",
        video_embeddings.shape
    )

    print(
        "Stage1 audio caption   :",
        final_audio_caption_embeddings.shape
    )

    print(
        "Stage1 video caption   :",
        final_video_caption_embeddings.shape
    )

    print(
        "Qwen3                  :",
        qwen3_embeddings.shape
    )

    print(
        "Mistral                :",
        mistral_embeddings.shape
    )

    print(
        "Gemma                  :",
        gemma_embeddings.shape
    )

    print("\nDone.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
