import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

OPENL3_DIR = "TAU-urban-audio-visual-scenes/create_data/numpy_features"
BERT_DIR = "bert_features"

TRAIN_CSV = "TAU-urban-audio-visual-scenes/create_data/evaluation_setup/fold1_train.csv"
EVAL_CSV = "TAU-urban-audio-visual-scenes/create_data/evaluation_setup/fold1_evaluate.csv"

RESULT_FILE = "svm_results.csv"


# ============================================================
# HELPER
# ============================================================

def normalize_name(name):
    """
    Convert filenames to a common representation.

    Examples:
        audio/airport-paris-7-351.wav
        video/airport-paris-7-351.mp4
        airport-paris-7-351

    -> airport-paris-7-351
    """
    name = os.path.basename(str(name))
    name = os.path.splitext(name)[0]
    return name


def align_features(features, feature_names, target_names, feature_name):
    """
    Check whether feature_names are already in target_names order.

    If aligned:
        return features unchanged.

    If not aligned:
        reorder features according to target_names.
    """

    feature_names = np.asarray([
        normalize_name(x) for x in feature_names
    ])

    target_names = np.asarray([
        normalize_name(x) for x in target_names
    ])

    print(f"\nChecking {feature_name}...")

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    if len(np.unique(feature_names)) != len(feature_names):
        raise ValueError(
            f"{feature_name}: duplicate filenames found."
        )

    # --------------------------------------------------------
    # Create mapping
    # --------------------------------------------------------

    name_to_idx = {
        name: i
        for i, name in enumerate(feature_names)
    }

    # --------------------------------------------------------
    # Check missing samples
    # --------------------------------------------------------

    missing = [
        name for name in target_names
        if name not in name_to_idx
    ]

    if missing:

        print(
            f"ERROR: {len(missing)} samples missing from "
            f"{feature_name}"
        )

        print("First missing samples:")

        for name in missing[:10]:
            print("  ", name)

        raise ValueError(
            f"{feature_name}: samples are missing."
        )

    # --------------------------------------------------------
    # Find required order
    # --------------------------------------------------------

    indices = np.array([
        name_to_idx[name]
        for name in target_names
    ])

    # --------------------------------------------------------
    # Check whether already aligned
    # --------------------------------------------------------

    already_aligned = np.array_equal(
        indices,
        np.arange(len(target_names))
    )

    if already_aligned:

        print(
            f"{feature_name}: ALREADY ALIGNED"
        )

        return features

    # --------------------------------------------------------
    # Reorder
    # --------------------------------------------------------

    print(
        f"{feature_name}: NOT ALIGNED"
    )

    print("Reordering features...")

    aligned_features = features[indices]

    print("Reordering completed.")

    return aligned_features


# ============================================================
# LOAD CSV FILES
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV,
    sep="\t"
)

eval_df = pd.read_csv(
    EVAL_CSV,
    sep="\t"
)

train_names = train_df["filename_audio"].values
eval_names = eval_df["filename_audio"].values

y_train = train_df["scene_label"].values
y_eval = eval_df["scene_label"].values


print("=" * 70)
print("DATASET")
print("=" * 70)

print("Train samples:", len(train_names))
print("Eval samples :", len(eval_names))


# ============================================================
# LOAD AUDIO
# ============================================================

audio_train_features = np.load(
    f"{OPENL3_DIR}/audio_tr_features.npy"
)

audio_train_names = np.load(
    f"{OPENL3_DIR}/audio_tr_names.npy",
    allow_pickle=True
)

audio_eval_features = np.load(
    f"{OPENL3_DIR}/audio_val_features.npy"
)

audio_eval_names = np.load(
    f"{OPENL3_DIR}/audio_val_names.npy",
    allow_pickle=True
)


print("\nAudio")
print("Train:", audio_train_features.shape)
print("Eval :", audio_eval_features.shape)


# ============================================================
# ALIGN AUDIO
# ============================================================

audio_train = align_features(
    audio_train_features,
    audio_train_names,
    train_names,
    "Audio Train"
)

audio_eval = align_features(
    audio_eval_features,
    audio_eval_names,
    eval_names,
    "Audio Eval"
)


# ============================================================
# LOAD VIDEO
# ============================================================

video_train_features = np.load(
    f"{OPENL3_DIR}/video_tr_features.npy"
)

video_train_names = np.load(
    f"{OPENL3_DIR}/video_tr_names.npy",
    allow_pickle=True
)

video_eval_features = np.load(
    f"{OPENL3_DIR}/video_val_features.npy"
)

video_eval_names = np.load(
    f"{OPENL3_DIR}/video_val_names.npy",
    allow_pickle=True
)


print("\nVideo")
print("Train:", video_train_features.shape)
print("Eval :", video_eval_features.shape)


# ============================================================
# ALIGN VIDEO
# ============================================================

video_train = align_features(
    video_train_features,
    video_train_names,
    train_names,
    "Video Train"
)

video_eval = align_features(
    video_eval_features,
    video_eval_names,
    eval_names,
    "Video Eval"
)


# ============================================================
# LOAD BERT EMBEDDINGS
# ============================================================

qwen_train_features = np.load(
    f"{BERT_DIR}/qwen_train.npy"
)

qwen_eval_features = np.load(
    f"{BERT_DIR}/qwen_eval.npy"
)

mistral_train_features = np.load(
    f"{BERT_DIR}/mistral_train.npy"
)

mistral_eval_features = np.load(
    f"{BERT_DIR}/mistral_eval.npy"
)

gemma_train_features = np.load(
    f"{BERT_DIR}/gemma_train.npy"
)

gemma_eval_features = np.load(
    f"{BERT_DIR}/gemma_eval.npy"
)


# ============================================================
# BERT NAMES
# ============================================================

bert_train_names = np.load(
    f"{BERT_DIR}/train_names.npy",
    allow_pickle=True
)

bert_eval_names = np.load(
    f"{BERT_DIR}/eval_names.npy",
    allow_pickle=True
)


print("\nBERT")
print("Qwen    :", qwen_train_features.shape,
      qwen_eval_features.shape)

print("Mistral :", mistral_train_features.shape,
      mistral_eval_features.shape)

print("Gemma   :", gemma_train_features.shape,
      gemma_eval_features.shape)


# ============================================================
# ALIGN QWEN
# ============================================================

qwen_train = align_features(
    qwen_train_features,
    bert_train_names,
    train_names,
    "Qwen Train"
)

qwen_eval = align_features(
    qwen_eval_features,
    bert_eval_names,
    eval_names,
    "Qwen Eval"
)


# ============================================================
# ALIGN MISTRAL
# ============================================================

mistral_train = align_features(
    mistral_train_features,
    bert_train_names,
    train_names,
    "Mistral Train"
)

mistral_eval = align_features(
    mistral_eval_features,
    bert_eval_names,
    eval_names,
    "Mistral Eval"
)


# ============================================================
# ALIGN GEMMA
# ============================================================

gemma_train = align_features(
    gemma_train_features,
    bert_train_names,
    train_names,
    "Gemma Train"
)

gemma_eval = align_features(
    gemma_eval_features,
    bert_eval_names,
    eval_names,
    "Gemma Eval"
)


# ============================================================
# FINAL ALIGNMENT CHECK
# ============================================================

assert audio_train.shape[0] == len(train_names)
assert audio_eval.shape[0] == len(eval_names)

assert video_train.shape[0] == len(train_names)
assert video_eval.shape[0] == len(eval_names)

assert qwen_train.shape[0] == len(train_names)
assert qwen_eval.shape[0] == len(eval_names)

assert mistral_train.shape[0] == len(train_names)
assert mistral_eval.shape[0] == len(eval_names)

assert gemma_train.shape[0] == len(train_names)
assert gemma_eval.shape[0] == len(eval_names)


print("\n" + "=" * 70)
print("FINAL ALIGNMENT")
print("=" * 70)

print("Audio   :", audio_train.shape, audio_eval.shape)
print("Video   :", video_train.shape, video_eval.shape)
print("Qwen    :", qwen_train.shape, qwen_eval.shape)
print("Mistral :", mistral_train.shape, mistral_eval.shape)
print("Gemma   :", gemma_train.shape, gemma_eval.shape)

print("\nAll features successfully aligned.")


# ============================================================
# FEATURE DICTIONARIES
# ============================================================

train_features = {
    "Audio": audio_train,
    "Video": video_train,
    "Qwen": qwen_train,
    "Mistral": mistral_train,
    "Gemma": gemma_train,
}

eval_features = {
    "Audio": audio_eval,
    "Video": video_eval,
    "Qwen": qwen_eval,
    "Mistral": mistral_eval,
    "Gemma": gemma_eval,
}


# ============================================================
# 8 CONFIGURATIONS
# ============================================================

configs = {

    "Audio": [
        "Audio"
    ],

    "Audio + Qwen": [
        "Audio",
        "Qwen"
    ],

    "Audio + Mistral": [
        "Audio",
        "Mistral"
    ],

    "Audio + Gemma": [
        "Audio",
        "Gemma"
    ],

    "Audio + Video": [
        "Audio",
        "Video"
    ],

    "Audio + Video + Qwen": [
        "Audio",
        "Video",
        "Qwen"
    ],

    "Audio + Video + Mistral": [
        "Audio",
        "Video",
        "Mistral"
    ],

    "Audio + Video + Gemma": [
        "Audio",
        "Video",
        "Gemma"
    ],
}


# ============================================================
# SVM TRAINING + EVALUATION
# ============================================================

results = []


for config_name, modalities in configs.items():

    print("\n" + "=" * 70)
    print(config_name)
    print("=" * 70)

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    X_train = np.concatenate(
        [
            train_features[m]
            for m in modalities
        ],
        axis=1
    )

    X_eval = np.concatenate(
        [
            eval_features[m]
            for m in modalities
        ],
        axis=1
    )

    print("Train shape:", X_train.shape)
    print("Eval shape :", X_eval.shape)

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            )
        )
    ])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = model.predict(
        X_eval
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_eval,
        y_pred
    )

    print(
        "Accuracy: {:.2f}%".format(
            accuracy * 100
        )
    )

    results.append({
        "Configuration": config_name,
        "Accuracy": accuracy,
        "Accuracy (%)": accuracy * 100,
        "Feature Dimension": X_train.shape[1]
    })


# ============================================================
# FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)


print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "Configuration",
            "Accuracy (%)",
            "Feature Dimension"
        ]
    ].to_string(
        index=False,
        formatters={
            "Accuracy (%)": "{:.2f}".format
        }
    )
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nResults saved to:", RESULT_FILE)
