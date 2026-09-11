import os

# ============================================================
# GPU 1
# ============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import clip
import torch
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

INPUT_CSV = "master_captions_qc.csv"

OUTPUT_CSV = "clip_scores.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"


# ============================================================
# VIDEO SAMPLING
# ============================================================

VIDEO_FPS = 2.0


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


valid = np.ones(len(df), dtype=bool)

for col in CAPTION_COLUMNS.values():
    valid &= df[col].apply(valid_caption).values


# Check files
valid &= df["filename_video"].apply(
    lambda x: os.path.isfile(
        os.path.join(DATA_ROOT, str(x))
    )
).values


eval_df = df.loc[valid].copy()

print("Common valid samples:", len(eval_df))


# ============================================================
# LOAD CLIP
# ============================================================

print("\nLoading CLIP...")

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=device,
)

clip_model.eval()

print("CLIP loaded.")


# ============================================================
# SAMPLE VIDEO FRAMES
# ============================================================

def sample_frames(video_path, fps=2.0):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if source_fps <= 0 or total_frames <= 0:
        cap.release()
        raise RuntimeError(
            f"Invalid video: {video_path}"
        )

    duration = total_frames / source_fps

    times = np.arange(
        0,
        duration,
        1.0 / fps,
    )

    frames = []

    for t in times:

        frame_idx = int(
            round(t * source_fps)
        )

        frame_idx = min(
            frame_idx,
            total_frames - 1,
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx,
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ============================================================
# CLIP STATISTICS
# ============================================================

def compute_clip(
    video_path,
    caption,
):

    frames = sample_frames(
        video_path,
        VIDEO_FPS,
    )

    if len(frames) == 0:
        raise RuntimeError(
            "No frames extracted."
        )

    # --------------------------------------------------------
    # Image batch
    # --------------------------------------------------------

    images = torch.stack(
        [
            preprocess(frame)
            for frame in frames
        ]
    ).to(device)

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = clip.tokenize(
        [caption],
        truncate=True,
    ).to(device)

    # --------------------------------------------------------
    # Encode
    # --------------------------------------------------------

    with torch.no_grad():

        image_features = (
            clip_model.encode_image(images)
        )

        text_features = (
            clip_model.encode_text(text)
        )
    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_features = (
        image_features
        / image_features.norm(
            dim=-1,
            keepdim=True,
        )
    )

    text_features = (
        text_features
        / text_features.norm(
            dim=-1,
            keepdim=True,
        )
    )
    # --------------------------------------------------------
    # Frame-level similarities
    # --------------------------------------------------------

    scores = (
        image_features
        @ text_features.T
    ).squeeze(1)

    scores = (
        scores
        .float()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p10": float(np.percentile(scores, 10)),
        "min": float(np.min(scores)),
        "num_frames": len(scores),
    }


# ============================================================
# EVALUATE
# ============================================================

results = []

print("\nStarting CLIP evaluation...")

for idx, row in tqdm(
    eval_df.iterrows(),
    total=len(eval_df),
    desc="CLIP",
):

    video_path = os.path.join(
        DATA_ROOT,
        str(row["filename_video"]),
    )

    result = {
        "scene_label": row["scene_label"],
        "filename_video": row["filename_video"],
    }

    for model, caption_col in CAPTION_COLUMNS.items():

        caption = str(
            row[caption_col]
        ).strip()

        try:

            scores = compute_clip(
                video_path,
                caption,
            )

            result[
                f"{model}_clip_mean"
            ] = scores["mean"]

            result[
                f"{model}_clip_std"
            ] = scores["std"]

            result[
                f"{model}_clip_p10"
            ] = scores["p10"]

            result[
                f"{model}_clip_min"
            ] = scores["min"]

            result[
                f"{model}_num_frames"
            ] = scores["num_frames"]

        except Exception as e:

            print(
                f"\nCLIP error "
                f"row={idx}, "
                f"model={model}: "
                f"{type(e).__name__}: {e}"
            )

            result[
                f"{model}_clip_mean"
            ] = np.nan

            result[
                f"{model}_clip_std"
            ] = np.nan

            result[
                f"{model}_clip_p10"
            ] = np.nan

            result[
                f"{model}_clip_min"
            ] = np.nan

            result[
                f"{model}_num_frames"
            ] = np.nan

    results.append(result)

    # Checkpoint every 100 samples
    if len(results) % 100 == 0:

        pd.DataFrame(results).to_csv(
            OUTPUT_CSV,
            index=False,
        )


# ============================================================
# SAVE
# ============================================================

scores = pd.DataFrame(results)

scores.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\nSaved:", OUTPUT_CSV)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLIP SUMMARY")
print("=" * 70)

for model in CAPTION_COLUMNS:

    print(f"\n{model.upper()}")

    print(
        "Mean:",
        scores[
            f"{model}_clip_mean"
        ].mean()
    )

    print(
        "P10:",
        scores[
            f"{model}_clip_p10"
        ].mean()
    )

    print(
        "Std:",
        scores[
            f"{model}_clip_std"
        ].mean()
    )
