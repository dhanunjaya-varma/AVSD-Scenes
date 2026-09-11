import torch
import torch.nn.functional as F
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FILES = {
    "STAGE1": "stage1_clap_embeddings_and_scores.pt",
    "QWEN3": "qwen3_clap_embeddings_and_scores.pt",
    "MISTRAL": "mistral_clap_embeddings_and_scores.pt",
    "GEMMA": "gemma_clap_embeddings_and_scores.pt",
}


OUTPUT_CSV = "clap_retrieval_results.csv"


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(similarity, k):
    """
    similarity:
        [N_queries, N_candidates]

    Ground-truth match:
        query i <-> candidate i
    """

    n = similarity.shape[0]

    _, topk = torch.topk(
        similarity,
        k=k,
        dim=1
    )

    targets = torch.arange(n).unsqueeze(1)

    correct = (topk == targets).any(dim=1)

    return correct.float().mean().item() * 100


# ============================================================
# EVALUATE ONE FILE
# ============================================================

def evaluate_file(model_name, filename):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    data = torch.load(
        filename,
        map_location="cpu"
    )

    print("Keys:", list(data.keys()))

    audio = data["audio_embeddings"].float()
    text = data["text_embeddings"].float()

    print("Audio embeddings:", audio.shape)
    print("Text embeddings :", text.shape)

    if audio.shape[0] != text.shape[0]:
        raise ValueError(
            f"Number of audio and text embeddings differ: "
            f"{audio.shape[0]} vs {text.shape[0]}"
        )

    # --------------------------------------------------------
    # Normalize CLAP embeddings
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Audio -> Text
    # --------------------------------------------------------

    print("\nComputing Audio → Text similarity matrix...")

    similarity = audio @ text.T

    a2t_r1 = recall_at_k(similarity, 1)
    a2t_r5 = recall_at_k(similarity, 5)
    a2t_r10 = recall_at_k(similarity, 10)

    print("\nAudio → Text")
    print(f"R@1  : {a2t_r1:.2f}")
    print(f"R@5  : {a2t_r5:.2f}")
    print(f"R@10 : {a2t_r10:.2f}")

    # --------------------------------------------------------
    # Text -> Audio
    # --------------------------------------------------------

    print("\nComputing Text → Audio similarity matrix...")

    similarity_t2a = similarity.T

    t2a_r1 = recall_at_k(similarity_t2a, 1)
    t2a_r5 = recall_at_k(similarity_t2a, 5)
    t2a_r10 = recall_at_k(similarity_t2a, 10)

    print("\nText → Audio")
    print(f"R@1  : {t2a_r1:.2f}")
    print(f"R@5  : {t2a_r5:.2f}")
    print(f"R@10 : {t2a_r10:.2f}")

    return {
        "Model": model_name,

        "Audio→Text R@1": a2t_r1,
        "Audio→Text R@5": a2t_r5,
        "Audio→Text R@10": a2t_r10,

        "Text→Audio R@1": t2a_r1,
        "Text→Audio R@5": t2a_r5,
        "Text→Audio R@10": t2a_r10,
    }


# ============================================================
# RUN ALL THREE
# ============================================================

results = []

for model_name, filename in FILES.items():

    result = evaluate_file(
        model_name,
        filename
    )

    results.append(result)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 100)
print("FINAL CLAP RETRIEVAL RESULTS")
print("=" * 100)

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
    f"\nSaved results to: {OUTPUT_CSV}"
)
