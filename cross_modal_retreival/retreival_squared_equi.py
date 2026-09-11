import torch
import torch.nn.functional as F
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "QWEN3": "qwen3_clap_embeddings_and_scores.pt",
    "MISTRAL": "mistral_clap_embeddings_and_scores.pt",
    "GEMMA": "gemma_clap_embeddings_and_scores.pt",

    # Uncomment if you have this file
    # "STAGE1": "stage1_clap_embeddings_and_scores.pt",
}

K_LIST = [1, 5, 10]

OUTPUT_CSV = "clap_instance_level_retrieval_results_squared_equi.csv"


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(topk_indices, k):
    """
    Exact instance-level Recall@K.

    Query i is correct if candidate i occurs
    in the top-k retrieved candidates.
    """

    targets = torch.arange(
        topk_indices.shape[0]
    ).unsqueeze(1)

    correct = (
        topk_indices[:, :k] == targets
    ).any(dim=1)

    return correct.float().mean().item() * 100


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, filename):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    data = torch.load(
        filename,
        map_location="cpu"
    )

    print("Keys:", list(data.keys()))

    audio = data["audio_embeddings"].float()
    text = data["text_embeddings"].float()

    print("Audio shape:", audio.shape)
    print("Text shape :", text.shape)

    if audio.shape[0] != text.shape[0]:
        raise ValueError(
            "Audio and text contain different "
            "numbers of samples."
        )

    N = audio.shape[0]

    # ========================================================
    # NORMALIZE EMBEDDINGS
    # ========================================================

    audio = F.normalize(
        audio,
        p=2,
        dim=1
    )

    text = F.normalize(
        text,
        p=2,
        dim=1
    )

    # ========================================================
    # FULL SQUARED EUCLIDEAN DISTANCE MATRIX
    # ========================================================

    print(
        f"Computing {N} x {N} distance matrix..."
    )

    dist = (
        audio.pow(2).sum(
            dim=1,
            keepdim=True
        )
        +
        text.pow(2).sum(
            dim=1,
            keepdim=True
        ).T
        -
        2 * audio @ text.T
    )

    # Numerical precision can occasionally produce
    # very small negative values.
    dist = torch.clamp(
        dist,
        min=0
    )

    # ========================================================
    # AUDIO -> TEXT
    # ========================================================

    a2t_topk = dist.topk(
        k=max(K_LIST),
        dim=1,
        largest=False
    ).indices

    print("\nAudio → Text")

    a2t_results = {}

    for k in K_LIST:

        r = recall_at_k(
            a2t_topk,
            k
        )

        a2t_results[k] = r

        print(
            f"R@{k:<2}: {r:.2f}"
        )

    # ========================================================
    # TEXT -> AUDIO
    # ========================================================

    t2a_topk = dist.T.topk(
        k=max(K_LIST),
        dim=1,
        largest=False
    ).indices

    print("\nText → Audio")

    t2a_results = {}

    for k in K_LIST:

        r = recall_at_k(
            t2a_topk,
            k
        )

        t2a_results[k] = r

        print(
            f"R@{k:<2}: {r:.2f}"
        )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "Model": model_name,

        "Audio→Text R@1":
            a2t_results[1],

        "Audio→Text R@5":
            a2t_results[5],

        "Audio→Text R@10":
            a2t_results[10],

        "Text→Audio R@1":
            t2a_results[1],

        "Text→Audio R@5":
            t2a_results[5],

        "Text→Audio R@10":
            t2a_results[10],
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

results = []

for model_name, filename in FILES.items():

    result = evaluate_model(
        model_name,
        filename
    )

    results.append(result)


# ============================================================
# FINAL TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

print("\n")
print("=" * 110)
print("INSTANCE-LEVEL CLAP RETRIEVAL")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"\nSaved to: {OUTPUT_CSV}"
)
