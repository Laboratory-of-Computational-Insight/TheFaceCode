# data_load.py
#
# Copyright notice: DISFA and FER2013 cannot be redistributed.
#   DISFA:   https://mohammadmahoor.com/pages/databases/disfa/
#   FER2013: https://www.kaggle.com/datasets/msambare/fer2013
#
# Expected directory layout (user must provide):
#
#   data/
#     DISFA_FEMALES/   *.mat files (one per female subject)
#     DISFA_MALES/     *.mat files (one per male subject)
#     disfa_labels.mat  — label matrix, columns: [index, identity, emotion, valence, arousal]
#     fer2013.mat       — keys: "data" (N,48,48), "labels" (N,6)
#                         label columns: [index, identity, emotion_classified, emotion_orig, valence, arousal]
#
# Edit the path constants below to match your local setup.

import os
import glob

import numpy as np
import torch
from scipy.io import loadmat

from objects.label_types import Attribute

# ── Path constants — edit these to match your local data layout ───────────────
DISFA_FEMALES_DIR = "./data/DISFA_FEMALES"   # folder of female-subject .mat files
DISFA_MALES_DIR   = "./data/DISFA_MALES"     # folder of male-subject .mat files
DISFA_LABELS_MAT  = "./data/disfa_labels.mat" # label matrix
FER2013_MAT       = "./data/fer2013.mat"      # FER2013 .mat file
# ─────────────────────────────────────────────────────────────────────────────

# Module-level cache used by train.py
DISFA  = None
data   = None
labels = None


def _load_disfa_mat_files(directory: str) -> np.ndarray:
    """Load and concatenate all .mat face arrays in a directory."""
    mats = sorted(glob.glob(os.path.join(directory, "*.mat")))
    if not mats:
        raise FileNotFoundError(
            f"No .mat files found in {directory!r}. "
            "Place your DISFA subject .mat files there."
        )
    arrays = []
    for path in mats:
        d = loadmat(path)
        # Each .mat file is expected to contain one variable with shape (N, H, W)
        # or (H, W, N) — adapt the key/transpose as needed for your export format.
        key = [k for k in d if not k.startswith("_")][0]
        arr = d[key]
        if arr.ndim == 3 and arr.shape[2] > arr.shape[0]:
            arr = arr.transpose(2, 0, 1)  # (H,W,N) → (N,H,W)
        arrays.append(arr)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def neg_labels(labels: torch.Tensor) -> None:
    """Flip negative valence labels to ensure consistent sign convention."""
    half_n       = len(labels) // 2
    other_half_n = len(labels) - half_n
    if 0 > labels[0, 1]:
        labels[:half_n, 1]  = -torch.zeros(half_n)
        labels[half_n:, 1]  = -torch.ones(other_half_n)


def disfa_train() -> dict:
    """
    Load DISFA faces for identity-loss training.

    Returns a dict mapping identity_id (int) → list of face tensors (1,H,W).

    Data comes from:
        DISFA_FEMALES_DIR / DISFA_MALES_DIR — raw face .mat files
        DISFA_LABELS_MAT                    — label matrix

    Label matrix columns: [index, identity, emotion, valence, arousal]
    """
    faces_raw = np.concatenate([
        _load_disfa_mat_files(DISFA_FEMALES_DIR),
        _load_disfa_mat_files(DISFA_MALES_DIR),
    ], axis=0)

    mat       = loadmat(DISFA_LABELS_MAT)
    label_arr = mat["labels"]  # shape (N, 5): [index, identity, emotion, valence, arousal]

    faces_tensor = torch.tensor(faces_raw).unsqueeze(1)  # (N,1,H,W)
    identity_ids = [int(row[1]) for row in label_arr]

    grouped: dict = {}
    for face, ide in zip(faces_tensor, identity_ids):
        grouped.setdefault(ide, []).append(face)
    return grouped


def disfa():
    """
    Load DISFA dataset for evaluation / analysis.

    Returns:
        faces_disfa : torch.Tensor of shape (N, 1, H, W), float32, pixel range [0,255]
        labels      : dict[int → tuple(identity, emotion, valence, arousal)]
                      • Attribute.IDENTITY = 0  (identity id)
                      • Attribute.EMOTION  = -3 (emotion class, capped at 6)
                      • Attribute.VALENCE  = -2
                      • Attribute.AROUSAL  = -1

    Data comes from:
        DISFA_FEMALES_DIR / DISFA_MALES_DIR — raw face .mat files
        DISFA_LABELS_MAT                    — label matrix

    Label matrix columns: [index, identity, emotion, valence, arousal]
    """
    faces1 = _load_disfa_mat_files(DISFA_FEMALES_DIR)
    faces2 = _load_disfa_mat_files(DISFA_MALES_DIR)

    mat       = loadmat(DISFA_LABELS_MAT)
    label_arr = mat["labels"]  # shape (N, 5)

    label_tensor = torch.tensor(label_arr, dtype=torch.float32)
    neg_labels(label_tensor)

    labels = {
        int(row[0]): (int(row[1]), int(row[2] if row[2] < 6 else 6), float(row[3]), float(row[4]))
        for row in label_tensor.numpy()
    }

    faces_raw    = np.vstack((faces1, faces2)).astype(np.float32)
    faces_tensor = torch.tensor(faces_raw).unsqueeze(1)  # (N,1,H,W)

    # 80 % random subset (matches original training convention)
    indexes = np.arange(len(faces_tensor))
    np.random.shuffle(indexes)
    indexes = sorted(indexes[:int(len(faces_tensor) * 0.8)])

    faces_tensor = faces_tensor[indexes]
    labels = {i: labels[old_i] for i, old_i in enumerate(indexes)}

    return faces_tensor, labels


def fer2013():
    """
    Load FER2013 dataset for evaluation / analysis.

    Returns:
        data   : torch.Tensor of shape (N, 1, 48, 48), float32, pixel range [0,1]
        labels : dict[int → tuple(identity, emotion_classified, emotion_orig, valence, arousal)]
                 • index 0 = identity
                 • index 1 = emotion_classified
                 • index 2 = emotion_orig
                 • index 3 = valence
                 • index 4 = arousal

    Data comes from FER2013_MAT.
    Expected keys in the .mat file:
        "data"   — array of shape (N, 48, 48), pixel values 0-255
        "labels" — array of shape (N, 6), columns: [index, identity, emotion_classified,
                   emotion_orig, valence, arousal]
    """
    if not os.path.exists(FER2013_MAT):
        raise FileNotFoundError(
            f"FER2013 .mat file not found at {FER2013_MAT!r}. "
            "Place your fer2013.mat there."
        )

    mat       = loadmat(FER2013_MAT)
    label_arr = torch.tensor(mat["labels"], dtype=torch.float32)
    neg_labels(label_arr)

    labels = {
        int(row[0]): (int(row[1]), int(row[2]), int(row[3]), float(row[4]), float(row[5]))
        for row in label_arr.numpy()
    }

    data = torch.tensor(mat["data"], dtype=torch.float32)
    if data.ndim == 3:
        data = data.unsqueeze(1)  # (N,48,48) → (N,1,48,48)

    return data / 255.0, labels


if __name__ == "__main__":
    faces, lbl = disfa()
    print(f"DISFA loaded: {faces.shape}, {len(lbl)} labels")
    d, lbl2 = fer2013()
    print(f"FER2013 loaded: {d.shape}, {len(lbl2)} labels")
