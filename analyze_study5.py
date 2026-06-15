"""
analyze_study5.py  —  All Study 5 post-training analyses.

Note: DISFA and FER2013 data cannot be shared due to copyright.
See data_load.py for the expected data format and path constants.

Runs three analyses in sequence on already-trained Study 5 models:

  C3.6  PCA sweep — how many top pixel-PCA components must be removed
        before identity SVM accuracy collapses to near-chance?

  C3.5  Identity SSC — silhouette score at P1/P3 for baseline vs.
        suppression model, with permutation null test.

  Valence  Valence/arousal R² for baseline vs. pixel_pca_k128 model,
        plus identyless_k128.pdf embedding scatter plot.

Usage:
    python analyze_study5.py [c36|c35|valence]   # one section only
    python analyze_study5.py                     # all three
"""

import os
import sys
import gc
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

from data_loader.fer2013_data_loader import FER2013DataLoader, Emotions
from data_load import disfa  # see data_load.py for expected format and path constants
from objects.context import get_context
from utils.model import create_model
from analyze.lower_dim import pca_lower_dim
from models.net import Net

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── which sections to run ─────────────────────────────────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("sections", nargs="*", default=["c36", "c35", "valence"])
args, _ = parser.parse_known_args()
RUN = set(args.sections)

context = get_context()
device  = context.device

# Output directory for PDF plots — edit to your LaTeX directory if needed
LATEX_DIR     = "./plots"

BASELINE_PATH = "./models/submitted/em3_normal/epoch99_last.pth"
SUPPRESS_PATH = "./models/submitted/em3_no_p1_iden_-0.25/epoch99_last.pth"
K128_PATH     = "./models/submitted/em3_no_raw_iden_k128/epoch99_last.pth"
LABELS_3      = [Emotions.HAPPINESS.value, Emotions.SADNESS.value, Emotions.NEUTRAL.value]


# ════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def remove_pixel_pca(images: torch.Tensor, pca_comp: np.ndarray, pca_mean: np.ndarray) -> torch.Tensor:
    """Mirror of train.py _remove_pixel_pca — project out top-k PCs per batch."""
    comp  = torch.from_numpy(pca_comp).float()
    mean  = torch.from_numpy(pca_mean).float()
    shape = images.shape
    flat  = images.reshape(len(images), -1) - mean
    proj  = flat @ comp.T @ comp
    return (flat - proj + mean).reshape(shape)


def extract_layer_embeddings(model, data: torch.Tensor, layer: str, n_pca: int,
                              pca_comp=None, pca_mean=None, batch_size: int = 256):
    """Run data through model, extract named layer, return PCA-reduced numpy array."""
    model.save_history_flag(True, elements=[layer])
    model.history.clear()
    model.eval()
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            if pca_comp is not None:
                batch = remove_pixel_pca(batch.cpu(), pca_comp, pca_mean).to(device)
            model(batch)
    acts = torch.cat([h[layer] for h in model.history], dim=0)
    model.history.clear()
    gc.collect()
    flat = acts.reshape(len(acts), -1).cpu()
    return pca_lower_dim(flat, n_components=n_pca, kernel_type="linear", degree=0)


def perm_ssc(embeddings, labels, n_perm: int = 100):
    ssc  = silhouette_score(embeddings, labels)
    rng  = np.random.default_rng(SEED)
    null = np.array([silhouette_score(embeddings, rng.permutation(labels)) for _ in range(n_perm)])
    z    = (ssc - null.mean()) / (null.std() + 1e-8)
    p    = (null >= ssc).mean()
    return ssc, z, p


def svm_kfold(X, y, k: int = 5):
    X = np.ascontiguousarray(X.astype(np.float32))
    y = np.asarray(y, dtype=np.int64)
    spl  = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=SEED)
    accs, baccs = [], []
    for tr, te in spl.split(X, y):
        clf = LinearSVC(random_state=SEED, dual="auto", max_iter=10_000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(baccs))


def valence_r2(embeddings: np.ndarray, valence, arousal, n_pca: int = 4):
    emb_t   = torch.tensor(embeddings)
    reduced = pca_lower_dim(emb_t, n_components=n_pca, kernel_type="linear", degree=0)
    v, a    = np.array(valence, dtype=float), np.array(arousal, dtype=float)
    r2_v    = float(Ridge(alpha=1.0).fit(reduced, v).score(reduced, v))
    r2_a    = float(Ridge(alpha=1.0).fit(reduced, a).score(reduced, a))
    return r2_v, r2_a


def remove_outliers_2pass(emb2d, valence):
    def one_pass(e, v):
        mask = np.ones(len(e), dtype=bool)
        for col in range(e.shape[1]):
            mask[e[:, col].argmax()] = False
            mask[e[:, col].argmin()] = False
        return e[mask], v[mask]
    emb2d, valence = one_pass(emb2d, valence)
    return one_pass(emb2d, valence)


def save_valence_plot(embeddings: np.ndarray, valence: np.ndarray, out_path: str):
    kpca = KernelPCA(n_components=4, kernel="linear")
    emb4 = kpca.fit_transform(embeddings)
    e2d, v = remove_outliers_2pass(emb4[:, :2], valence)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(e2d[:, 0], e2d[:, 1], c=v, cmap="jet", s=18, alpha=0.65, linewidths=0)
    ax.set_xlabel("PC1", fontsize=38, fontweight="bold", labelpad=12)
    ax.set_ylabel("PC2", fontsize=38, fontweight="bold", labelpad=12)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_linewidth(2.5)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Plot saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
#  C3.6  —  Pixel PCA sweep: how many dims to remove until SVM collapses
# ════════════════════════════════════════════════════════════════════════════
if "c36" in RUN:
    print("\n" + "="*70)
    print("C3.6  Pixel PCA sweep — identity SVM accuracy vs. k")
    print("="*70)

    print("Loading DISFA (pixel matrices) ...")
    faces_disfa, labels_dict = disfa()

    N = len(faces_disfa)
    if N == 0:
        print("  [stub] No DISFA data available — skipping C3.6.")
    else:
        identity_labels = np.array([labels_dict[i][0] for i in range(N)])
        X_full = faces_disfa.reshape(N, -1).float().numpy().astype(np.float32)
        n_ids  = len(np.unique(identity_labels))
        chance = 1.0 / n_ids
        print(f"  {N} frames, {X_full.shape[1]} pixels, {n_ids} identities  (chance={chance:.4f})")

        MAX_DIM = min(N - 1, X_full.shape[1], 512)
        print(f"Fitting PCA({MAX_DIM}) on DISFA pixels ...")
        pca_disfa = PCA(n_components=MAX_DIM, random_state=SEED)
        pca_disfa.fit(X_full)
        cumvar = np.cumsum(pca_disfa.explained_variance_ratio_)

        K_VALUES = [k for k in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] if k < MAX_DIM]
        print(f"\n{'k':>6} {'var_removed':>12} {'SVM acc':>8} {'Bal acc':>8}")
        print("-"*42)
        sweep_results = []
        for k in K_VALUES:
            if k == 0:
                X_in = X_full
                vr   = 0.0
            else:
                scores = (X_full - pca_disfa.mean_) @ pca_disfa.components_[:k].T
                X_in   = X_full - (scores @ pca_disfa.components_[:k] + pca_disfa.mean_) + pca_disfa.mean_
                vr     = cumvar[k - 1]
            acc, std, bacc = svm_kfold(X_in, identity_labels)
            sweep_results.append((k, vr, acc, bacc))
            tag = " <-- near chance" if acc < 0.10 else (" <-- COLLAPSE" if acc < 0.30 else "")
            print(f"{k:>6} {vr*100:>11.1f}% {acc:>8.4f} {bacc:>8.4f}{tag}")

        del faces_disfa, X_full, pca_disfa
        gc.collect()

    print("\nC3.6 done.")


# ════════════════════════════════════════════════════════════════════════════
#  C3.5  —  Identity SSC at P1/P3: baseline vs. suppression model
# ════════════════════════════════════════════════════════════════════════════
if "c35" in RUN:
    print("\n" + "="*70)
    print("C3.5  Identity SSC at P1/P3 — baseline vs. suppression model")
    print("="*70)

    print("Loading DISFA ...")
    faces, labels_dict = disfa()

    N = len(faces)
    if N == 0:
        print("  [stub] No DISFA data available — skipping C3.5.")
    else:
        identity_labels_c35 = np.array([labels_dict[i][0] for i in range(N)])
        print(f"  {N} frames, {len(np.unique(identity_labels_c35))} identities")

        LAYERS = [
            ("batch_norm1", 4),   # P1
            ("dropout_4",   2),   # P3
        ]

        ssc_results = {}
        for label, model_path in [("baseline", BASELINE_PATH), ("suppression", SUPPRESS_PATH)]:
            print(f"\n  [{label}]")
            try:
                model, _ = create_model(context, 3, Net, model_path, device)
            except Exception as e:
                print(f"    ERROR: {e}")
                ssc_results[label] = None
                continue
            model.eval()
            ssc_results[label] = {}
            for layer_name, n_pca in LAYERS:
                emb = extract_layer_embeddings(model, faces.to(device), layer_name, n_pca)
                ssc, z, p = perm_ssc(emb, identity_labels_c35)
                ssc_results[label][layer_name] = (ssc, z, p)
                print(f"    {layer_name}: SSC={ssc:.4f}  z={z:.2f}  p={p:.4f}")
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del faces, labels_dict
        gc.collect()

        print("\nC3.5 SUMMARY")
        print(f"  {'Layer':<22} {'Baseline':>26} {'Suppression':>26}")
        print(f"  {'-'*76}")
        for lname, _ in LAYERS:
            tag = "P1 (batch_norm1)" if lname == "batch_norm1" else "P3 (dropout_4)"

            def fmt(d, k):
                t = d.get(k, {}).get(lname) if d else None
                return "N/A" if t is None else f"{t[0]:.4f} (z={t[1]:.2f}, p={t[2]:.4f})"

            print(
                f"  {tag:<22} {fmt(ssc_results.get('baseline'), 'baseline'):<26} "
                f"{fmt(ssc_results.get('suppression'), 'suppression'):<26}"
            )

    print("\nC3.5 done.")


# ════════════════════════════════════════════════════════════════════════════
#  Valence  —  R² + embedding plot: baseline vs. pixel_pca_k128
# ════════════════════════════════════════════════════════════════════════════
if "valence" in RUN:
    print("\n" + "="*70)
    print("Valence  R² comparison — baseline vs. pixel_pca_k128")
    print("="*70)

    # Fit PCA(128) on FER2013 training images — same preprocessing as k=128 training
    print("Fitting PCA(128) on FER2013 training images ...")
    fer13 = FER2013DataLoader()
    train_dl, _, _ = fer13.get_train_val_test(64, labels=LABELS_3)
    all_imgs = []
    for batch_imgs, _ in train_dl:
        all_imgs.append(batch_imgs.float().reshape(len(batch_imgs), -1).numpy())
    all_imgs = np.vstack(all_imgs)
    pca128   = PCA(n_components=128, random_state=SEED)
    pca128.fit(all_imgs)
    pca_comp = pca128.components_
    pca_mean = pca128.mean_
    print(f"  Top-128 PCs explain {pca128.explained_variance_ratio_.sum()*100:.1f}% of FER2013 variance.")
    del all_imgs
    gc.collect()

    print("Loading DISFA ...")
    faces, labels_dict = disfa()

    N = len(faces)
    if N == 0:
        print("  [stub] No DISFA data available — skipping Valence section.")
    else:
        valence_all = np.array([labels_dict[i][-2] for i in range(N)])
        arousal_all = np.array([labels_dict[i][-1] for i in range(N)])
        valid       = (np.abs(valence_all) <= 1) & (np.abs(arousal_all) <= 1)
        faces_v     = faces[valid]
        val_v       = valence_all[valid]
        aro_v       = arousal_all[valid]
        print(f"  {valid.sum()} frames with valid valence/arousal")

        MODELS = [
            ("baseline",       BASELINE_PATH, None),
            ("pixel_pca_k128", K128_PATH,     (pca_comp, pca_mean)),
        ]
        val_results = {}
        for model_name, model_path, pca_args in MODELS:
            print(f"\n  [{model_name}]  Loading {model_path} ...")
            try:
                model, _ = create_model(context, 3, Net, model_path, device)
            except Exception as e:
                print(f"    ERROR: {e}")
                val_results[model_name] = None
                continue
            model.eval()

            comp_arg = pca_args[0] if pca_args else None
            mean_arg = pca_args[1] if pca_args else None

            model.save_history_flag(True, elements=["dropout_4"])
            model.history.clear()
            with torch.no_grad():
                for i in range(0, len(faces_v), 256):
                    batch = faces_v[i:i+256].to(device)
                    if comp_arg is not None:
                        batch = remove_pixel_pca(batch.cpu(), comp_arg, mean_arg).to(device)
                    model(batch)
            acts = torch.cat([h["dropout_4"] for h in model.history], dim=0)
            model.history.clear()
            gc.collect()
            emb = acts.reshape(len(acts), -1).cpu().numpy()
            print(f"    Embeddings: {emb.shape}")

            r2_v, r2_a = valence_r2(emb, val_v, aro_v, n_pca=4)
            val_results[model_name] = dict(r2_v=r2_v, r2_a=r2_a)
            print(f"    Valence R²: {r2_v:.4f}   Arousal R²: {r2_a:.4f}")

            if model_name == "pixel_pca_k128":
                os.makedirs(LATEX_DIR, exist_ok=True)
                out_pdf = os.path.join(LATEX_DIR, "identyless_k128.pdf")
                print("    Generating embedding plot ...")
                save_valence_plot(emb, val_v, out_pdf)

            del model, acts, emb
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del faces, labels_dict
        gc.collect()

        print("\nValence SUMMARY TABLE")
        print(f"  {'Model':<25} {'Valence R²':>12} {'Arousal R²':>12}")
        print(f"  {'-'*52}")
        for name, res in val_results.items():
            if res is None:
                print(f"  {name:<25} {'ERROR':>12}")
            else:
                print(f"  {name:<25} {res['r2_v']:>12.4f} {res['r2_a']:>12.4f}")

    print("\nValence done.")


print("\n" + "="*70)
print("All requested sections complete.")
print("="*70)
