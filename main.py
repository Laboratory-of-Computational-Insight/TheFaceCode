"""
main.py — Train all Study 5 model variants.

Usage:
    python main.py

Models are saved to SAVE_PATH/<task_name>/epoch<N>_last.pth.
Set SAVE_PATH and RESTORE_PATH in config.yaml (model.save_path / model.restore_path)
or override the constants below.
"""

import os

import numpy as np
from sklearn.decomposition import PCA

from data_loader.fer2013_data_loader import FER2013DataLoader, Emotions

from config import configuration
from objects.context import get_context
from train.train import Train, IdentityLossConfig, IdentityLossMode

from models.net_smaller import Net as SmallerNet
from models.net import Net


def get7labels():
    return [
        Emotions.HAPPINESS.value, Emotions.SADNESS.value, Emotions.NEUTRAL.value,
        Emotions.ANGER.value, Emotions.FEAR.value, Emotions.DISGUST.value,
        Emotions.SURPRISE.value,
    ]


def get3labels():
    return [Emotions.HAPPINESS.value, Emotions.SADNESS.value, Emotions.NEUTRAL.value]


def train(
    batch_size, save_path, task_name, restore_path,
    labels_size=3, model_type=Net,
    identity_loss=None, remove_iden=False,
    pixel_pca_k=0, pixel_pca_components=None, pixel_pca_mean=None,
):
    assert labels_size in [3, 7], f"labels_size must be 3 or 7, got {labels_size}"
    assert model_type in [Net, SmallerNet], f"model_type must be Net or SmallerNet, got {model_type}"

    labels = get3labels() if labels_size == 3 else get7labels()

    fer13 = FER2013DataLoader()
    train_dl, val_dl, test_dl = fer13.get_train_val_test(batch_size, labels=labels)

    train_run = Train(
        get_context(),
        train=train_dl,
        val=val_dl,
        test=test_dl,
        iter_limit=1_000_000,
        n_classes=len(labels),
        epochs=100,
        log=50,
        save_path=os.path.join(save_path, task_name),
        restore_path=restore_path,
        net_type=model_type,
        identity_loss=identity_loss,
        remove_iden=remove_iden,
        pixel_pca_k=pixel_pca_k,
        pixel_pca_components=pixel_pca_components,
        pixel_pca_mean=pixel_pca_mean,
    )
    train_run.run()


def eval(test=None, path=None, model=None, labels_size=3, net_type=None, remove_iden=None):
    assert labels_size in [3, 7], f"labels_size must be 3 or 7, got {labels_size}"

    if test is None:
        fer13  = FER2013DataLoader()
        labels = get3labels() if labels_size == 3 else get7labels()
        _, _, test = fer13.get_train_val_test(1, labels=labels)

    train_run = Train(
        get_context(),
        test=test,
        iter_limit=1_000_000,
        n_classes=labels_size,
        epochs=100,
        log=50,
        restore_path=path,
        net_type=net_type,
        model=model,
        remove_iden=remove_iden,
    )
    metrics = train_run.test()
    print(metrics)


def fit_pixel_pca(batch_size, k):
    """Fit PCA(k) on FER2013 training images; return (components, mean)."""
    print(f"Fitting PCA({k}) on FER2013 training images ...")
    fer13  = FER2013DataLoader()
    labels = get3labels()
    train_dl, _, _ = fer13.get_train_val_test(batch_size, labels=labels)
    all_imgs = []
    for batch_imgs, _ in train_dl:
        all_imgs.append(batch_imgs.float().reshape(len(batch_imgs), -1).numpy())
    all_imgs = np.vstack(all_imgs)
    pca = PCA(n_components=k, random_state=42)
    pca.fit(all_imgs)
    print(f"  Done. Top-{k} PCs explain {pca.explained_variance_ratio_.sum()*100:.1f}% of training variance.")
    return pca.components_, pca.mean_


# ── training variants ─────────────────────────────────────────────────────────
# Each entry: (task_name, labels_size, model_type, identity_loss_cfg, remove_iden, pixel_pca_k)
VARIANTS = [
    # Baselines
    ("em3_normal",            3, Net,        None,                                                False, 0),
    ("em7_normal",            7, Net,        None,                                                False, 0),
    ("em3_smaller",           3, SmallerNet, None,                                                False, 0),
    # Study 5 identity-loss variants (alternating mode)
    ("em3_no_p1_iden_-0.25",  3, Net,        IdentityLossConfig.suppress_identity_p1_alternating(), False, 0),
    ("em3_yes_p3_iden",       3, Net,        IdentityLossConfig.retain_identity_p3_alternating(),   False, 0),
    # Pixel-space PCA removal — k=16
    ("em3_no_raw_iden",       3, Net,        None,                                                True,  0),
    # Condition (4): force emotion clustering at P3 (no DISFA needed)
    ("em3_no_p3_dimensional", 3, Net,        IdentityLossConfig.force_emotion_clustering_p3(),   False, 0),
    ("em7_no_p3_dimensional", 7, Net,        IdentityLossConfig.force_emotion_clustering_p3(),   False, 0),
    # Pixel-space PCA removal — k=128 sweep (C3.6)
    ("em3_no_raw_iden_k128",  3, Net,        None,                                                False, 128),
    # Mixed-minibatch variants (reviewer C3.4)
    ("em3_yes_p3_iden_mixed", 3, Net,        IdentityLossConfig.retain_identity_p3_mixed(),        False, 0),
    ("em3_no_p1_iden_mixed",  3, Net,        IdentityLossConfig.suppress_identity_p1_mixed(),      False, 0),
]

BATCH_SIZE   = configuration.data.batch_size
SAVE_PATH    = configuration.model.save_path    # e.g. "./models"
RESTORE_PATH = configuration.model.restore_path  # e.g. None or a checkpoint path

# Pre-fit PCA(128) once if any variant needs it
_pca128_comp, _pca128_mean = None, None
if any(v[5] == 128 for v in VARIANTS):
    _pca128_comp, _pca128_mean = fit_pixel_pca(BATCH_SIZE, 128)

for task_name, label_size, model_cls, cfg, remove_iden, pixel_pca_k in VARIANTS:
    print(f"---------------{task_name}---------------------")

    pca_comp = _pca128_comp if pixel_pca_k == 128 else None
    pca_mean = _pca128_mean if pixel_pca_k == 128 else None

    train(
        BATCH_SIZE, SAVE_PATH, task_name, RESTORE_PATH,
        labels_size=label_size,
        model_type=model_cls,
        identity_loss=cfg,
        remove_iden=remove_iden,
        pixel_pca_k=pixel_pca_k,
        pixel_pca_components=pca_comp,
        pixel_pca_mean=pca_mean,
    )
    eval(
        path=os.path.join(SAVE_PATH, task_name, "epoch99_last.pth"),
        labels_size=label_size,
        net_type=model_cls,
        remove_iden=remove_iden,
    )
