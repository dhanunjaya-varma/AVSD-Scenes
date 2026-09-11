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

OUTPUT_CSV = "clap_scene_retrieval_results.csv"


# ============================================================
# RECALL@K - SCENE LEVEL
# ============================================================

def scene_recall_at_k(
    similarity,
    query_scene_labels,
    candidate_scene_labels,
    k
):
    """
    Scene-level Recall@K.

    A query is counted as correct if ANY of the top-k
    retrieved candidates has the same scene label.

    Parameters
    ----------
    similarity:
        [N_queries, N_candidates]

    query_scene_labels:
        List of scene labels for queries.

    candidate_scene_labels:
        List of scene labels for candidates.

    k:
        Recall@K

    Returns
    -------
    recall percentage
    """

    n_queries = similarity.shape[0]

    # Top-k candidate indices
    _, topk_indices = torch.topk(
        similarity,
        k=k,
        dim=1
    )

    correct = 0

    for i in range(n_queries):

        query_scene = query_scene_labels[i]

        retrieved_indices = topk_indices[i]

        # Check whether ANY retrieved item has
        # the same scene
        for idx in retrieved_indices:

            candidate_scene = candidate_scene_labels[
                idx.item()
            ]

            if candidate_scene == query_scene:

                correct += 1
                break

    return (correct / n_queries) * 100


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, filename):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = torch.load(
        filename,
        map_location="cpu"
    )

    print("Keys:")
    for key in data.keys():
        print(" ", key)

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    audio = data["audio_embeddings"].float()

    text = data["text_embeddings"].float()

    # --------------------------------------------------------
    # Scene labels
    #
    # Change this key if your file uses another name.
    # --------------------------------------------------------

    scene_labels = data["scene_label"]

    # Convert tensor -> Python list if necessary
    if torch.is_tensor(scene_labels):
        scene_labels = scene_labels.tolist()

    scene_labels = [
        str(x) for x in scene_labels
    ]

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    assert len(scene_labels) == audio.shape[0], (
        f"Number of scene labels ({len(scene_labels)}) "
        f"does not match audio embeddings "
        f"({audio.shape[0]})"
    )

    assert len(scene_labels) == text.shape[0], (
        f"Number of scene labels ({len(scene_labels)}) "
        f"does not match text embeddings "
        f"({text.shape[0]})"
    )

    print("\nNumber of samples:", len(scene_labels))
    print(
        "Number of scenes:",
        len(set(scene_labels))
    )

    # --------------------------------------------------------
    # Normalize
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
    # Similarity matrix
    # --------------------------------------------------------

    print("\nComputing similarity matrix...")

    similarity = audio @ text.T

    # ========================================================
    # AUDIO -> TEXT
    # ========================================================

    print("\nAudio → Text")

    a2t_r1 = scene_recall_at_k(
        similarity,
        scene_labels,
        scene_labels,
        1
    )

    a2t_r5 = scene_recall_at_k(
        similarity,
        scene_labels,
        scene_labels,
        5
    )

    a2t_r10 = scene_recall_at_k(
        similarity,
        scene_labels,
        scene_labels,
        10
    )

    print(f"R@1  : {a2t_r1:.2f}")
    print(f"R@5  : {a2t_r5:.2f}")
    print(f"R@10 : {a2t_r10:.2f}")

    # ========================================================
    # TEXT -> AUDIO
    # ========================================================

    print("\nText → Audio")

    similarity_t2a = similarity.T

    t2a_r1 = scene_recall_at_k(
        similarity_t2a,
        scene_labels,
        scene_labels,
        1
    )

    t2a_r5 = scene_recall_at_k(
        similarity_t2a,
        scene_labels,
        scene_labels,
        5
    )

    t2a_r10 = scene_recall_at_k(
        similarity_t2a,
        scene_labels,
        scene_labels,
        10
    )

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
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 110)
print("FINAL SCENE-LEVEL CLAP RETRIEVAL RESULTS")
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
    f"\nResults saved to: {OUTPUT_CSV}"
)
