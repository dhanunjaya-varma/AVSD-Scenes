import torch
import torch.nn.functional as F
import pandas as pd


# ============================================================
# CONFIG
# ============================================================


OUTPUT_CSV = "imagebind_retrieval_results.csv"


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

def evaluate_file(model_name, audio, text):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

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

data = torch.load("imagebind_embeddings_new.pt", map_location="cpu")

audio = data["audio_embeddings"].float()
stage1_text = data["final_audio_caption_embeddings"].float()
qwen_text = data["qwen3_text_embeddings"].float()
mistral_text = data["mistral_text_embeddings"].float()
gemma_text = data["gemma_text_embeddings"].float()

result1 = evaluate_file('STAGE1',audio, stage1_text)
result2 = evaluate_file('QWEN3',audio, qwen_text)
result3 = evaluate_file('MISTRAL',audio, mistral_text)
result4 = evaluate_file('GEMMA', audio, gemma_text)

results.append(result1)
results.append(result2)
results.append(result3)
results.append(result4)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 100)
print("FINAL IMAGEBIND RETRIEVAL RESULTS")
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
