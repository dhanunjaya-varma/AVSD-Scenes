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

    # Uncomment if available
    # "STAGE1": "stage1_clap_embeddings_and_scores.pt",
}

K_LIST = [1, 5, 10]

OUTPUT_CSV = "clap_scene_level_retrieval_results_squared_equi.csv"


# ============================================================
# FIND SCENE LABEL KEY
# ============================================================

def get_scene_labels(data):

    possible_keys = [
        "scene_labels",
        "scene_label",
        "scene_names",
        "scene_name",
    ]

    for key in possible_keys:

        if key in data:

            scenes = data[key]

            if torch.is_tensor(scenes):
                scenes = scenes.tolist()

            return [
                str(x)
                for x in scenes
            ]

    raise KeyError(
        "Could not find scene labels. "
        "Expected one of: "
        + str(possible_keys)
    )


# ============================================================
# SCENE RECALL@K
# ============================================================

def scene_recall_at_k(
    topk_indices,
    query_scenes,
    candidate_scenes,
    k
):

    correct = 0

    for i in range(
        len(query_scenes)
    ):

        query_scene = query_scenes[i]

        retrieved_indices = (
            topk_indices[i, :k]
            .tolist()
        )

        retrieved_scenes = [
            candidate_scenes[j]
            for j in retrieved_indices
        ]

        # At least one retrieved candidate
        # has the same scene.
        if query_scene in retrieved_scenes:

            correct += 1

    return (
        correct
        / len(query_scenes)
        * 100
    )


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

    audio = data[
        "audio_embeddings"
    ].float()

    text = data[
        "text_embeddings"
    ].float()

    scenes = get_scene_labels(
        data
    )

    print(
        "Audio shape:",
        audio.shape
    )

    print(
        "Text shape :",
        text.shape
    )

    print(
        "Number of scenes:",
        len(set(scenes))
    )

    N = audio.shape[0]

    if len(scenes) != N:

        raise ValueError(
            f"Number of scene labels "
            f"({len(scenes)}) does not match "
            f"number of samples ({N})."
        )

    if text.shape[0] != N:

        raise ValueError(
            "Audio and text have "
            "different numbers of samples."
        )

    # ========================================================
    # NORMALIZE
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
    # FULL SQUARED EUCLIDEAN DISTANCE
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

        r = scene_recall_at_k(
            a2t_topk,
            query_scenes=scenes,
            candidate_scenes=scenes,
            k=k
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

        r = scene_recall_at_k(
            t2a_topk,
            query_scenes=scenes,
            candidate_scenes=scenes,
            k=k
        )

        t2a_results[k] = r

        print(
            f"R@{k:<2}: {r:.2f}"
        )

    # ========================================================
    # RETURN
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
print("SCENE-LEVEL CLAP RETRIEVAL")
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
