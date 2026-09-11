import os
import torch
import clip
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV = "master_captions_qc.csv"

DATA_ROOT = "/home/dhanunjaya/scratch/TAU"

OUTPUT_FILE = "clip_video_caption_embeddings_and_scores.pt"

VIDEO_FPS = 2.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64


# ============================================================
# LOAD CLIP
# ============================================================

print("=" * 70)
print("Loading CLIP")
print("=" * 70)

clip_model, preprocess = clip.load(
    "ViT-B/32",
    device=DEVICE,
)

clip_model.eval()

print("Device:", DEVICE)
print("Model :", "ViT-B/32")


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("\nNumber of samples:", len(df))

CAPTION_COLUMNS = [
    "qwen3_caption",
    "mistral_caption",
    "gemma_caption",
    "final_video_caption",
]

required_columns = [
    "filename_video",
] + CAPTION_COLUMNS

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Missing column: {col}"
        )


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(
    video_path,
    fps=2.0
):
    """
    Extract frames from a video at approximately
    the specified FPS.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():

        print(
            f"WARNING: Cannot open video: {video_path}"
        )

        return []

    original_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if original_fps <= 0:

        cap.release()

        return []

    duration = (
        total_frames / original_fps
    )

    sample_interval = 1.0 / fps

    timestamps = np.arange(
        0,
        duration,
        sample_interval
    )

    frames = []

    for timestamp in timestamps:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        ret, frame = cap.read()

        if not ret:
            continue

        # BGR -> RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frames.append(
            Image.fromarray(frame)
        )

    cap.release()

    return frames


# ============================================================
# VIDEO -> CLIP EMBEDDING
# ============================================================

@torch.no_grad()
def get_video_embedding(
    video_path,
    fps=2.0
):

    frames = extract_video_frames(
        video_path,
        fps
    )

    if len(frames) == 0:

        return None

    frame_embeddings = []

    # --------------------------------------------------------
    # Process frames in batches
    # --------------------------------------------------------

    for start in range(
        0,
        len(frames),
        BATCH_SIZE
    ):

        batch_frames = frames[
            start:start + BATCH_SIZE
        ]

        images = torch.stack(
            [
                preprocess(frame)
                for frame in batch_frames
            ]
        ).to(DEVICE)

        embeddings = clip_model.encode_image(
            images
        )

        # Normalize each frame
        embeddings = embeddings / (
            embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        frame_embeddings.append(
            embeddings
        )

    frame_embeddings = torch.cat(
        frame_embeddings,
        dim=0
    )

    # --------------------------------------------------------
    # Mean pooling over frames
    # --------------------------------------------------------

    video_embedding = frame_embeddings.mean(
        dim=0
    )

    # Normalize final video embedding
    video_embedding = video_embedding / (
        video_embedding.norm()
    )

    return video_embedding.cpu()


# ============================================================
# TEXT -> CLIP EMBEDDING
# ============================================================

@torch.no_grad()
def get_text_embeddings(
    captions
):

    embeddings = []

    for start in range(
        0,
        len(captions),
        BATCH_SIZE
    ):

        batch = captions[
            start:start + BATCH_SIZE
        ]

        # Handle missing captions
        batch = [
            "" if pd.isna(x)
            else str(x)
            for x in batch
        ]

        tokens = clip.tokenize(
            batch,
            truncate=True
        ).to(DEVICE)

        text_embeddings = clip_model.encode_text(
            tokens
        )

        # Normalize
        text_embeddings = (
            text_embeddings
            / text_embeddings.norm(
                dim=-1,
                keepdim=True
            )
        )

        embeddings.append(
            text_embeddings.cpu()
        )

    return torch.cat(
        embeddings,
        dim=0
    )


# ============================================================
# EXTRACT VIDEO EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("Extracting VIDEO CLIP embeddings")
print("=" * 70)

video_embeddings = []

valid_indices = []

for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Videos"
):

    video_file = row[
        "filename_video"
    ]

    video_path = os.path.join(
        DATA_ROOT,
        video_file
    )

    embedding = get_video_embedding(
        video_path,
        fps=VIDEO_FPS
    )

    if embedding is None:

        print(
            f"\nSkipping sample {idx}: "
            f"{video_path}"
        )

        continue

    video_embeddings.append(
        embedding
    )

    valid_indices.append(
        idx
    )


# ============================================================
# ALIGN DATA
# ============================================================

print(
    "\nValid videos:",
    len(video_embeddings)
)

if len(video_embeddings) == 0:

    raise RuntimeError(
        "No video embeddings were extracted."
    )

video_embeddings = torch.stack(
    video_embeddings
)

df_valid = df.iloc[
    valid_indices
].reset_index(drop=True)


print(
    "Video embeddings:",
    video_embeddings.shape
)


# ============================================================
# EXTRACT CAPTION EMBEDDINGS
# ============================================================

caption_embeddings = {}


for column in CAPTION_COLUMNS:

    print("\n" + "=" * 70)
    print(
        f"Extracting CLIP embeddings: {column}"
    )
    print("=" * 70)

    embeddings = get_text_embeddings(
        df_valid[column].tolist()
    )

    caption_embeddings[column] = embeddings

    print(
        f"{column}:",
        embeddings.shape
    )


# ============================================================
# COMPUTE PAIRED VIDEO-CAPTION SIMILARITY
# ============================================================

print("\n" + "=" * 70)
print("Computing paired video-caption similarity")
print("=" * 70)

similarities = {}

for column in CAPTION_COLUMNS:

    text_embeddings = (
        caption_embeddings[column]
    )

    # Since both embeddings are L2-normalized,
    # dot product = cosine similarity.

    similarity_matrix = (
        video_embeddings
        @ text_embeddings.T
    )

    # Diagonal = corresponding
    # video-caption pairs

    pair_similarity = (
        similarity_matrix.diag()
    )

    similarities[column] = (
        pair_similarity
    )


# ============================================================
# PRINT STATISTICS
# ============================================================

print("\n")
print("=" * 90)
print("VIDEO-CAPTION CLIP SIMILARITY")
print("=" * 90)

print(
    f"{'Caption':<25}"
    f"{'Mean':>12}"
    f"{'P10':>12}"
    f"{'Std':>12}"
)

print("-" * 90)


for column in CAPTION_COLUMNS:

    scores = similarities[column].float()

    mean = scores.mean().item()

    p10 = torch.quantile(
        scores,
        0.10
    ).item()

    std = scores.std().item()

    print(
        f"{column:<25}"
        f"{mean:>12.4f}"
        f"{p10:>12.4f}"
        f"{std:>12.4f}"
    )


# ============================================================
# SAVE
# ============================================================

output = {

    # --------------------------------------------------------
    # Video embeddings
    # --------------------------------------------------------

    "video_embeddings":
        video_embeddings,

    # --------------------------------------------------------
    # Caption embeddings
    # --------------------------------------------------------

    "qwen3_embeddings":
        caption_embeddings[
            "qwen3_caption"
        ],

    "mistral_embeddings":
        caption_embeddings[
            "mistral_caption"
        ],

    "gemma_embeddings":
        caption_embeddings[
            "gemma_caption"
        ],

    "final_video_caption_embeddings":
        caption_embeddings[
            "final_video_caption"
        ],

    # --------------------------------------------------------
    # Paired similarities
    # --------------------------------------------------------

    "qwen3_similarity":
        similarities[
            "qwen3_caption"
        ],

    "mistral_similarity":
        similarities[
            "mistral_caption"
        ],

    "gemma_similarity":
        similarities[
            "gemma_caption"
        ],

    "final_video_caption_similarity":
        similarities[
            "final_video_caption"
        ],

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    "scene_labels":
        df_valid[
            "scene_label"
        ].tolist()
        if "scene_label" in df_valid.columns
        else None,

    "filename_video":
        df_valid[
            "filename_video"
        ].tolist(),

    "qwen3_caption":
        df_valid[
            "qwen3_caption"
        ].tolist(),

    "mistral_caption":
        df_valid[
            "mistral_caption"
        ].tolist(),

    "gemma_caption":
        df_valid[
            "gemma_caption"
        ].tolist(),

    "final_video_caption":
        df_valid[
            "final_video_caption"
        ].tolist(),

    "clip_model":
        "ViT-B/32",

    "video_sampling_fps":
        VIDEO_FPS,

    "embedding_dimension":
        video_embeddings.shape[1],
}


torch.save(
    output,
    OUTPUT_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)

print(
    "Video embeddings:",
    video_embeddings.shape
)

for column in CAPTION_COLUMNS:

    print(
        f"{column:<25}:",
        caption_embeddings[column].shape
    )

print(
    "\nOutput file:",
    OUTPUT_FILE
)
