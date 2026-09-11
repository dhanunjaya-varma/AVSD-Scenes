import os

# ============================================================
# GPU 1
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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

OUTPUT_CSV = "clap_scores.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"


# ============================================================
# AUDIO SETTINGS
# ============================================================

TARGET_SR = 48000

WINDOW_SECONDS = 5.0

HOP_SECONDS = 2.5


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
# CAPTION COLUMNS
# ============================================================

CAPTION_COLUMNS = {
    "qwen3": "qwen3_caption",
    "mistral": "mistral_caption",
    "gemma": "gemma_caption",
}


# ============================================================
# COMMON VALID SUBSET
# ============================================================

def valid_caption(x):

    if pd.isna(x):
        return False

    return len(str(x).strip()) > 0


valid = np.ones(
    len(df),
    dtype=bool,
)


for col in CAPTION_COLUMNS.values():

    valid &= (
        df[col]
        .apply(valid_caption)
        .values
    )


valid &= df["filename_audio"].apply(
    lambda x: os.path.isfile(
        os.path.join(
            DATA_ROOT,
            str(x),
        )
    )
).values


eval_df = df.loc[
    valid
].copy()


print(
    "Common valid samples:",
    len(eval_df)
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
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(x):

    x = np.asarray(x)

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return x / np.maximum(
        norm,
        1e-12,
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b,
):

    a = normalize_embedding(a)

    b = normalize_embedding(b)

    return float(
        np.sum(
            a * b,
            axis=1,
        )[0]
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(
    audio_path,
):

    audio, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True,
    )

    return audio.astype(
        np.float32
    )


# ============================================================
# GLOBAL CLAP
# ============================================================

def compute_global_clap(
    audio,
    caption,
):

    audio = audio.reshape(
        1,
        -1,
    )

    audio_embedding = (
        clap_model
        .get_audio_embedding_from_data(
            x=audio,
            use_tensor=False,
        )
    )

    text_embedding = (
        clap_model
        .get_text_embedding(
            [caption],
            use_tensor=False,
        )
    )

    return cosine_similarity(
        audio_embedding,
        text_embedding,
    )


# ============================================================
# WINDOWED CLAP
# ============================================================

def compute_windowed_clap(
    audio,
    caption,
):

    window_samples = int(
        WINDOW_SECONDS
        * TARGET_SR
    )

    hop_samples = int(
        HOP_SECONDS
        * TARGET_SR
    )

    audio_length = len(audio)

    # --------------------------------------------------------
    # Short audio
    # --------------------------------------------------------

    if audio_length <= window_samples:

        windows = [
            audio
        ]

    else:

        windows = []

        start = 0

        while (
            start
            + window_samples
            <= audio_length
        ):

            windows.append(
                audio[
                    start:
                    start + window_samples
                ]
            )

            start += hop_samples

        # ----------------------------------------------------
        # Include final segment if not already included
        # ----------------------------------------------------

        if start < audio_length:

            final_window = audio[
                max(
                    0,
                    audio_length
                    - window_samples,
                ):
            ]

            windows.append(
                final_window
            )

    # --------------------------------------------------------
    # Text embedding
    # --------------------------------------------------------

    text_embedding = (
        clap_model
        .get_text_embedding(
            [caption],
            use_tensor=False,
        )
    )

    # --------------------------------------------------------
    # Compute CLAP for each window
    # --------------------------------------------------------

    scores = []

    for window in windows:

        window = window.reshape(
            1,
            -1,
        )

        audio_embedding = (
            clap_model
            .get_audio_embedding_from_data(
                x=window,
                use_tensor=False,
            )
        )

        score = cosine_similarity(
            audio_embedding,
            text_embedding,
        )

        scores.append(score)

    scores = np.asarray(
        scores,
        dtype=np.float32,
    )

    return {
        "mean": float(
            np.mean(scores)
        ),
        "std": float(
            np.std(scores)
        ),
        "p10": float(
            np.percentile(
                scores,
                10,
            )
        ),
        "min": float(
            np.min(scores)
        ),
        "num_windows": len(scores),
    }


# ============================================================
# EVALUATION
# ============================================================

results = []

print("\nStarting CLAP evaluation...")

for idx, row in tqdm(
    eval_df.iterrows(),
    total=len(eval_df),
    desc="CLAP",
):

    audio_path = os.path.join(
        DATA_ROOT,
        str(row["filename_audio"]),
    )

    result = {

        "scene_label":
            row["scene_label"],

        "filename_audio":
            row["filename_audio"],
    }


    # ========================================================
    # LOAD AUDIO ONCE
    # ========================================================

    try:

        audio = load_audio(
            audio_path
        )

    except Exception as e:

        print(
            f"\nAudio error "
            f"row={idx}: "
            f"{type(e).__name__}: {e}"
        )

        continue


    # ========================================================
    # EACH MODEL
    # ========================================================

    for model, caption_col in (
        CAPTION_COLUMNS.items()
    ):

        caption = str(
            row[caption_col]
        ).strip()


        # ----------------------------------------------------
        # GLOBAL CLAP
        # ----------------------------------------------------

        try:

            global_score = (
                compute_global_clap(
                    audio,
                    caption,
                )
            )

            result[
                f"{model}_clap_global"
            ] = global_score

        except Exception as e:

            print(
                f"\nGlobal CLAP error "
                f"row={idx}, "
                f"model={model}: "
                f"{type(e).__name__}: {e}"
            )

            result[
                f"{model}_clap_global"
            ] = np.nan


        # ----------------------------------------------------
        # WINDOWED CLAP
        # ----------------------------------------------------

        try:

            window_scores = (
                compute_windowed_clap(
                    audio,
                    caption,
                )
            )

            result[
                f"{model}_clap_window_mean"
            ] = window_scores["mean"]

            result[
                f"{model}_clap_window_std"
            ] = window_scores["std"]

            result[
                f"{model}_clap_window_p10"
            ] = window_scores["p10"]

            result[
                f"{model}_clap_window_min"
            ] = window_scores["min"]

            result[
                f"{model}_clap_num_windows"
            ] = window_scores[
                "num_windows"
            ]

        except Exception as e:

            print(
                f"\nWindowed CLAP error "
                f"row={idx}, "
                f"model={model}: "
                f"{type(e).__name__}: {e}"
            )

            result[
                f"{model}_clap_window_mean"
            ] = np.nan

            result[
                f"{model}_clap_window_std"
            ] = np.nan

            result[
                f"{model}_clap_window_p10"
            ] = np.nan

            result[
                f"{model}_clap_window_min"
            ] = np.nan

            result[
                f"{model}_clap_num_windows"
            ] = np.nan


    results.append(result)


    # ========================================================
    # CHECKPOINT
    # ========================================================

    if len(results) % 100 == 0:

        pd.DataFrame(
            results
        ).to_csv(
            OUTPUT_CSV,
            index=False,
        )


# ============================================================
# SAVE
# ============================================================

scores = pd.DataFrame(
    results
)

scores.to_csv(
    OUTPUT_CSV,
    index=False,
)


print(
    "\nSaved:",
    OUTPUT_CSV
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CLAP SUMMARY"
)

print(
    "=" * 70
)


for model in CAPTION_COLUMNS:

    print(
        f"\n{model.upper()}"
    )

    print(
        "Global CLAP:",
        scores[
            f"{model}_clap_global"
        ].mean()
    )

    print(
        "Window mean:",
        scores[
            f"{model}_clap_window_mean"
        ].mean()
    )

    print(
        "Window P10:",
        scores[
            f"{model}_clap_window_p10"
        ].mean()
    )

    print(
        "Window std:",
        scores[
            f"{model}_clap_window_std"
        ].mean()
    )
