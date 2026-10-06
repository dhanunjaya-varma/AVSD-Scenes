import os
import h5py
import numpy as np
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

AUDIO_H5 = "features_data/audio_features_data/tr.hdf5"
VIDEO_H5 = "features_data/video_features_data/tr.hdf5"

OUTPUT_DIR = "numpy_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONVERT HDF5 TO NUMPY
# ============================================================

def convert_hdf5(h5_path, modality):

    features = []
    labels = []
    names = []

    with h5py.File(h5_path, "r") as f:

        for label in tqdm(f.keys(), desc=modality):

            # 0/audio or 0/video
            modality_group = f[label][modality]

            for name in modality_group.keys():

                # e.g. airport-paris-7-351
                sample_group = modality_group[name]

                # 0, 1, 2, ..., 74
                frame_keys = sorted(
                    sample_group.keys(),
                    key=int
                )

                # [num_frames, 512]
                frame_features = np.stack(
                    [sample_group[k][()] for k in frame_keys]
                )

                # [512]
                feature = frame_features.mean(axis=0)

                features.append(feature)
                labels.append(int(label))
                names.append(name)

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    names = np.asarray(names)

    np.save(
        os.path.join(OUTPUT_DIR, f"{modality}_tr_features.npy"),
        features
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{modality}_tr_labels.npy"),
        labels
    )

    np.save(
        os.path.join(OUTPUT_DIR, f"{modality}_tr_names.npy"),
        names
    )

    print(f"\n{modality}:")
    print("Features:", features.shape)
    print("Labels:  ", labels.shape)
    print("Names:   ", names.shape)


# ============================================================
# AUDIO
# ============================================================

convert_hdf5(AUDIO_H5, "audio")


# ============================================================
# VIDEO
# ============================================================

convert_hdf5(VIDEO_H5, "video")
