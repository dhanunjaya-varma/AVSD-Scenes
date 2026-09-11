import torch
import torch.nn.functional as F
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

OUTPUT_CSV = "imagebind_scene_retrieval_results.csv"


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

def evaluate_file(model_name, data, audio, text):

    print("\n" + "=" * 80)
    print(model_name)
    print("=" * 80)

    # --------------------------------------------------------
    # Scene labels
    #
    # Change this key if your file uses another name.
    # --------------------------------------------------------

    scene_labels = data["scene_labels"]

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

    print("\nVideo → Text")

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

    print("\nText → Video")

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

        "Video→Text R@1": a2t_r1,
        "Video→Text R@5": a2t_r5,
        "Video→Text R@10": a2t_r10,

        "Text→Video R@1": t2a_r1,
        "Text→Video R@5": t2a_r5,
        "Text→Video R@10": t2a_r10,
    }


# ============================================================
# RUN ALL MODELS
# ============================================================

results = []

data = torch.load("imagebind_embeddings_new.pt", map_location="cpu")

audio = data["video_embeddings"].float()
stage1_text = data["final_video_caption_embeddings"].float()
qwen_text = data["qwen3_text_embeddings"].float()
mistral_text = data["mistral_text_embeddings"].float()
gemma_text = data["gemma_text_embeddings"].float()

result1 = evaluate_file('STAGE1', data, audio, stage1_text)
result2 = evaluate_file('QWEN3', data, audio, qwen_text)
result3 = evaluate_file('MISTRAL', data, audio, mistral_text)
result4 = evaluate_file('GEMMA', data, audio, gemma_text)

results.append(result1)
results.append(result2)
results.append(result3)
results.append(result4)


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n")
print("=" * 110)
print("FINAL SCENE-LEVEL IMAGEBIND RETRIEVAL RESULTS")
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
