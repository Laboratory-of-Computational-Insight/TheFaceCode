"""
main_plot.py — Post-training analysis and embedding visualisation.

Runs silhouette score, SVM identity classification, cross-validated identity
SVM (sv_fold), valence/arousal R² orthogonality, and 2-D embedding plots for
each (dataset, model, layer, attribute) combination defined below.

Usage:
    python main_plot.py
"""

import gc
from builtins import enumerate
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.feature_selection import f_regression
import statsmodels.api as sm
from torch import Tensor

from sklearn.preprocessing import StandardScaler

from analyze.lower_dim import pca_lower_dim, pca_get_dim, cumsum_index

from data_loader.fer2013_data_loader import Emotions, seven_to_three_conversion

from data_load import disfa, fer2013
from models.net import Net
from models.net_smaller import Net as NetSmaller
from objects.context import get_context
from objects.label_types import Attribute
from utils.cs_plot.cs_plot import plot_embedding, remove_outlayers

from utils.model import create_model
from collections import defaultdict

from sklearn.metrics import silhouette_score
import torch

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    top_k_accuracy_score,
)


# ── context / globals ─────────────────────────────────────────────────────────
context = get_context()

layer_attribute_param = {}
regression_results    = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def get_all_emotions_elements():
    return list(Emotions)


def get_dummy_conversion():
    conversion = {e.value: e.value for e in Emotions}
    conversion[7] = 1
    return conversion


# ── model factories ───────────────────────────────────────────────────────────

def create_model3(context, i=0):
    file = "./models/submitted/em3_normal/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions", model, model, emotions, conversion, elements


def create_model3_no_p1_iden(context, i=0):
    file = "./models/submitted/em3_no_p1_iden/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions_no_p1_iden", model, model, emotions, conversion, elements


def create_model3_yes_p3_iden(context, i=0):
    file = "./models/submitted/em3_yes_p3_iden/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions_yes_p3_iden", model, model, emotions, conversion, elements


def create_model3_no_raw_iden(context, i=0):
    file = "./models/submitted/em3_no_raw_iden/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions_no_raw_iden", model, model, emotions, conversion, elements


def _fit_pixel_pca_128():
    """Fit PCA(128) on FER2013 training images (same as during k=128 training)."""
    from data_loader.fer2013_data_loader import FER2013DataLoader, Emotions as _Emotions
    labels3 = [_Emotions.HAPPINESS.value, _Emotions.SADNESS.value, _Emotions.NEUTRAL.value]
    fer13 = FER2013DataLoader()
    train_dl, _, _ = fer13.get_train_val_test(64, labels=labels3)
    all_imgs = []
    for batch_imgs, _ in train_dl:
        all_imgs.append(batch_imgs.float().reshape(len(batch_imgs), -1).numpy())
    all_imgs = np.vstack(all_imgs)
    pca = PCA(n_components=128, random_state=42)
    pca.fit(all_imgs)
    print(f"  PCA(128) fitted: top-128 PCs explain "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% of FER2013 variance.")
    return (torch.from_numpy(pca.components_).float(),
            torch.from_numpy(pca.mean_).float())


def create_model3_no_raw_iden_k128(context, i=0):
    """k=128 pixel-PCA-removal model (Study 5 C3.6 sweep endpoint)."""
    file = "./models/submitted/em3_no_raw_iden_k128/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()

    pca_comp, pca_mean = _fit_pixel_pca_128()

    def _remove_pixel_pca(images: torch.Tensor) -> torch.Tensor:
        shape = images.shape
        flat  = images.reshape(len(images), -1) - pca_mean
        proj  = flat @ pca_comp.T @ pca_comp
        return (flat - proj + pca_mean).reshape(shape)

    original_forward = model.forward

    def forward_with_pca_removal(x):
        x = _remove_pixel_pca(x.cpu()).to(x.device)
        return original_forward(x)

    model.forward = forward_with_pca_removal

    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions_no_raw_iden_k128", model, model, emotions, conversion, elements


def create_model3_no_p3_dim(context, i=0):
    file = "./models/submitted/em3_no_p3_dimensional/epoch99_last.pth"
    print(file)
    model, _ = create_model(context, 3, Net, file, context.device)
    model.eval()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   4, model.dropout_4),
    ]
    emotions   = [Emotions.HAPPINESS, Emotions.SADNESS, Emotions.NEUTRAL]
    conversion = seven_to_three_conversion
    return "3_emotions_no_p3_dim", model, model, emotions, conversion, elements


def create_model7_60(context):
    model, _ = create_model(
        context, 7, Net, "./models/submitted/em7_normal/epoch99_last.pth", context.device
    )
    model.eval()
    emotions   = get_all_emotions_elements()
    conversion = get_dummy_conversion()
    elements = [
        ("batch_norm1", 4, model.batch_norm1),
        ("flatten",     4, model.flatten),
        ("dropout_4",   2, model.dropout_4),
    ]
    return "7_emotions", model, model, emotions, conversion, elements


# optional comparison models — uncomment to include:
# def create_model_rmn(context): ...   # ResidualMaskingNetwork
# def create_model_emofan(context): .. # EmoNet / emofan
# def create_model_vgg(context): ...   # VGG face emotion


# ── analysis helpers ──────────────────────────────────────────────────────────

def compute_valence_arousal_orthogonality(
    embeddings,
    valence,
    arousal,
    emo_labels_pred,
    emo_labels_gt=None,
    alpha: float = 1.0,
    pca_components: int | None = None,
    min_samples_per_emo: int = 15,
    indent: str = "\t\t",
):
    X   = np.asarray(embeddings)
    y_v = np.asarray(valence, dtype=float)
    y_a = np.asarray(arousal, dtype=float)
    emo_pred = np.asarray(emo_labels_pred)
    emo_gt   = None if emo_labels_gt is None else np.asarray(emo_labels_gt)

    N, D = X.shape
    if pca_components is None:
        pca_components = min(2, N, D)

    pca = PCA(n_components=pca_components, svd_solver="full", random_state=0)
    Z   = pca.fit_transform(X)

    reg_v = Ridge(alpha=alpha).fit(Z, y_v)
    reg_a = Ridge(alpha=alpha).fit(Z, y_a)
    w_v, w_a = reg_v.coef_, reg_a.coef_
    denom     = np.linalg.norm(w_v) * np.linalg.norm(w_a) + 1e-8
    global_cos = float(np.dot(w_v, w_a) / denom)
    r2_v = float(reg_v.score(Z, y_v))
    r2_a = float(reg_a.score(Z, y_a))
    print(f"{indent}VA global: cos={global_cos:.3f}, R2_v={r2_v:.3f}, R2_a={r2_a:.3f}")

    v_hat = reg_v.predict(Z)
    a_hat = reg_a.predict(Z)
    w_vn  = w_v / (np.linalg.norm(w_v) + 1e-8)
    w_an  = w_a / (np.linalg.norm(w_a) + 1e-8)

    def _per_emotion_block(emo, tag: str):
        per_cos, wv_list, wa_list, n_list = [], [], [], []
        print(f"{indent}--- Per-emotion ({tag}) ---")
        for e in np.unique(emo):
            m   = (emo == e)
            n_e = int(m.sum())
            if n_e < min_samples_per_emo:
                continue
            reg_v_e = Ridge(alpha=alpha).fit(Z[m], y_v[m])
            reg_a_e = Ridge(alpha=alpha).fit(Z[m], y_a[m])
            wv_e, wa_e = reg_v_e.coef_, reg_a_e.coef_
            denom_e    = np.linalg.norm(wv_e) * np.linalg.norm(wa_e) + 1e-8
            cos_e      = float(np.dot(wv_e, wa_e) / denom_e)
            per_cos.append(cos_e); wv_list.append(wv_e); wa_list.append(wa_e); n_list.append(n_e)
            print(
                f"{indent}Emotion {int(e)}: "
                f"V={v_hat[m].mean():.2f}±{v_hat[m].std():.2f}, "
                f"A={a_hat[m].mean():.2f}±{a_hat[m].std():.2f}, "
                f"cos(V,A)={cos_e:.3f} (n={n_e})"
            )
        if not per_cos:
            return
        per_cos = np.asarray(per_cos)
        n_arr   = np.asarray(n_list, dtype=float)
        print(f"{indent}{tag} per-emotion: mean cos={per_cos.mean():.3f}, median={np.median(per_cos):.3f}")
        wv_normed = np.stack([w / (np.linalg.norm(w) + 1e-8) for w in wv_list])
        wa_normed = np.stack([w / (np.linalg.norm(w) + 1e-8) for w in wa_list])
        wv_overall = (n_arr[:, None] * wv_normed).sum(axis=0)
        wa_overall = (n_arr[:, None] * wa_normed).sum(axis=0)
        denom_tag  = np.linalg.norm(wv_overall) * np.linalg.norm(wa_overall) + 1e-8
        cos_raw    = float(np.dot(wv_overall, wa_overall) / denom_tag)
        angle_deg  = float(np.degrees(np.arccos(np.clip(abs(cos_raw), -1.0, 1.0))))
        print(f"{indent}{tag} overall vector: cos_raw={cos_raw:.3f}, angle={angle_deg:.2f}°")

    _per_emotion_block(emo_pred, "predicted")
    if emo_gt is not None:
        _per_emotion_block(emo_gt, "ground-truth")


def svm_identity(h_embedings, colors_h):
    """One-vs-rest SVM on identity labels; results stored in layer_attribute_param."""
    X = np.asarray(h_embedings)
    y = np.asarray(colors_h)

    unique_classes = np.unique(y)
    if unique_classes.shape[0] < 2 or X.shape[0] < 10:
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42,
        stratify=y if unique_classes.shape[0] > 1 else None,
    )

    svm_ovr = OneVsRestClassifier(
        LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=42)
    )
    svm_ovr.fit(X_train, y_train)
    y_pred = svm_ovr.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred, labels=unique_classes)

    topk = {}
    try:
        y_score  = svm_ovr.decision_function(X_test)
        n_classes = unique_classes.shape[0]
        for k in (1, 3, 5):
            if k <= n_classes:
                topk[k] = top_k_accuracy_score(y_test, y_score, k=k, labels=unique_classes)
    except Exception:
        pass

    print("\t\t\t\t\tSVM(OVR) identity metrics")
    print("\t\t\t\t\t\tacc", acc, "balanced_acc", bacc, "topk", topk)
    print("\t\t\t\t\t\treport\n", classification_report(y_test, y_pred, digits=4))
    print("\t\t\t\t\t\tconfusion_matrix shape", cm.shape)

    lap = layer_attribute_param
    lap[element][attribute]["svm_acc"]  = lap[element][attribute].get("svm_acc",  [])
    lap[element][attribute]["svm_bacc"] = lap[element][attribute].get("svm_bacc", [])
    lap[element][attribute]["svm_acc"].append(acc)
    lap[element][attribute]["svm_bacc"].append(bacc)
    for k_key in (1, 3, 5):
        if k_key in topk:
            key = f"svm_top{k_key}"
            lap[element][attribute][key] = lap[element][attribute].get(key, [])
            lap[element][attribute][key].append(topk[k_key])


def sv_fold(faces_disfa: Tensor, labels: dict[Any, Any]):
    """Stratified 10-fold cross-validated identity SVM."""
    if isinstance(faces_disfa, torch.Tensor):
        X = faces_disfa.detach().to("cpu").reshape(faces_disfa.shape[0], -1).numpy()
    else:
        X = np.asarray(faces_disfa).reshape(len(faces_disfa), -1)
    X = np.ascontiguousarray(X, dtype=np.float32)

    first = labels[0] if isinstance(labels, dict) else labels[0]
    if np.isscalar(first):
        y = np.asarray([labels[i] for i in range(len(labels))], dtype=np.int64)
    elif isinstance(first, (list, tuple, np.ndarray)) and len(first) > 0 and np.isscalar(first[0]):
        y = np.asarray([labels[i][0] for i in range(len(labels))], dtype=np.int64)
    else:
        y = np.asarray([labels[i][Attribute.IDENTITY] for i in range(len(labels))], dtype=np.int64)

    k       = 10
    splitter = StratifiedShuffleSplit(n_splits=k, test_size=0.2, random_state=0)
    accs, baccs = [], []
    for fold, (tr, te) in enumerate(splitter.split(X, y), start=1):
        clf = LinearSVC(random_state=0, dual="auto", max_iter=10_000)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))
        print(f"split {fold:02d}/{k}: acc={accs[-1]:.4f} bacc={baccs[-1]:.4f}")
    accs, baccs = np.array(accs), np.array(baccs)
    print(
        f"{k}x 80/20: acc mean={accs.mean():.4f} std={accs.std(ddof=1):.4f} | "
        f"bacc mean={baccs.mean():.4f} std={baccs.std(ddof=1):.4f}"
    )


# ── datasets and models to analyse ───────────────────────────────────────────
# Load datasets once
datas_labels = [
    ["disfa",   *disfa()],
    ["fer2013", *fer2013()],
]

# Activate the model factories you want to run (uncomment as needed):
models = [
    create_model3(context, 1),
    # create_model3_no_raw_iden(context, 1),
    # create_model3_no_raw_iden_k128(context, 1),
    # create_model3_no_p1_iden(context, 1),
    # create_model3_yes_p3_iden(context, 1),
    # create_model3_no_p3_dim(context, 1),
    # create_model7_60(context),
]

# ── main analysis loop ────────────────────────────────────────────────────────
print("starting")

for data_name, data_orig, labels_orig in datas_labels:
    print("\t", data_name)

    if "fer" in data_name:
        limit = 1000
    else:
        limit = 13_000

    MAX_PER_ID = 1000
    id_counts  = defaultdict(int)
    index_map  = []
    for idx, label in labels_orig.items():
        id_temp = label[0]
        id_counts[id_temp] += 1
        if id_counts[id_temp] < MAX_PER_ID:
            index_map.append(idx)

    data_orig   = data_orig[index_map]
    labels_orig = {new_i: labels_orig[old_i] for new_i, old_i in enumerate(index_map)}
    rand_index_pre_change = torch.arange(len(data_orig))

    for model_name, model, model_f, emotions, conversion, elements_corolation in models:
        print("\t\t", model_name)

        rand_index = rand_index_pre_change

        model.history.clear()
        gc.collect()

        data   = data_orig[rand_index]
        labels = {new_i: labels_orig[i.item()] for new_i, i in enumerate(rand_index)}

        valid_indices = [
            i for i in range(len(data))
            if labels[i][Attribute.EMOTION] in emotions
        ]
        data        = data[valid_indices]
        labels_temp = {}
        emo_dist    = {e: 0 for e in emotions}
        for i in valid_indices:
            label = list(labels[i])
            emo   = label[Attribute.EMOTION]
            emo_dist[emo] += 1
            label[Attribute.EMOTION] = conversion[emo]
            labels_temp[len(labels_temp)] = tuple(label)
        labels = labels_temp

        if hasattr(model, "save_history_flag"):
            model.save_history_flag(True, elements=[e[0] for e in elements_corolation])

        data        = data.to(context.device)
        predictions = []

        for i in range(len(data)):
            instance   = data[i:i + 1]
            prediction = model_f(instance)
            predictions.append(prediction.argmax().item())

        print("\t\t\t", "accuracy",
              (np.array([l[Attribute.EMOTION] for l in labels.values()]) == np.array(predictions)).mean())

        for element, percent_index_80, sub_model in elements_corolation:
            print("\t\t\t", element)
            intermidate_predictions = torch.cat([h[element] for h in model.history], dim=0)

            for h in model.history:
                del h[element]

            for attribute in [
                Attribute.EMOTION,
                Attribute.VALENCE,
                Attribute.AROUSAL,
                Attribute.IDENTITY,
            ]:
                gc.collect()
                print("\t\t\t\t", attribute)

                colors = [labels[i][attribute] for i in range(len(intermidate_predictions))]
                if attribute == Attribute.EMOTION:
                    colors = [predictions[i] for i in range(len(intermidate_predictions))]

                va_groups = [
                    ("valence",  [labels[i][Attribute.VALENCE] for i in range(len(intermidate_predictions))]),
                    ("arousal",  [labels[i][Attribute.AROUSAL] for i in range(len(intermidate_predictions))]),
                    ("emo_gt",   [labels[i][Attribute.EMOTION]  for i in range(len(intermidate_predictions))]),
                    ("emo_pred", [int(p.item()) if hasattr(p, "item") else int(p) for p in predictions]),
                ]

                flattened = intermidate_predictions.reshape(len(intermidate_predictions), -1)
                h_index, h_value, l_index, l_value = pca_get_dim(flattened, 0.85, n_components=128)
                h_embedings = pca_lower_dim(intermidate_predictions, n_components=h_index)

                h_embedings, colors_h, preds_h, va_groups = remove_outlayers(h_embedings, colors, predictions, va_groups)
                h_embedings, colors_h, preds_h, va_groups = remove_outlayers(h_embedings, colors_h, preds_h, va_groups)

                valence_f  = np.array(va_groups[0][1], dtype=float)
                arousal_f  = np.array(va_groups[1][1], dtype=float)
                emo_gt_f   = np.array(va_groups[2][1])
                emo_pred_f = np.array(va_groups[3][1])

                if attribute == Attribute.IDENTITY:
                    svm_identity(h_embedings, colors_h)
                    sv_fold(h_embedings, colors_h)

                if attribute == Attribute.VALENCE:
                    compute_valence_arousal_orthogonality(
                        embeddings=h_embedings,
                        valence=valence_f,
                        arousal=arousal_f,
                        emo_labels_pred=emo_pred_f,
                        emo_labels_gt=emo_gt_f,
                        alpha=1.0,
                    )

                sil_h = (
                    silhouette_score(h_embedings, colors_h)
                    if attribute in (Attribute.EMOTION, Attribute.IDENTITY)
                    else -1
                )
                print("\t\t\t\t\tsil", sil_h)

                if attribute == Attribute.IDENTITY and element == "dropout_4":
                    pca = PCA(n_components=min(intermidate_predictions.shape))
                    pca.fit(intermidate_predictions.to("cpu").reshape(len(intermidate_predictions), -1))
                    index, value, minus_one_index, minus_one_value = cumsum_index(
                        0.85, pca.explained_variance_ratio_
                    )
                    lower = (
                        (intermidate_predictions
                         - torch.tensor(pca.mean_).to(intermidate_predictions.device))
                        @ torch.tensor(pca.components_[index:].T).to(intermidate_predictions.device)
                    )
                    lower, colors_h, preds_h, _ = remove_outlayers(lower, colors, predictions)
                    lower, colors_h, preds_h, _ = remove_outlayers(lower, colors_h, preds_h)
                    sil_of_iden_inverse = silhouette_score(lower, colors_h)
                    print("\t\t\t\t\t inverse sil", sil_of_iden_inverse)

                layer_attribute_param.setdefault(element, {})
                layer_attribute_param[element].setdefault(attribute, {})
                layer_attribute_param[element][attribute].setdefault("silhouette", []).append(sil_h)

                flattened = intermidate_predictions.reshape(len(intermidate_predictions), -1)
                pca10     = pca_lower_dim(flattened, n_components=10)

                x    = pca10[:, 0]
                y    = pca10[:, 1]
                xys  = np.array([x, y]).T
                scaler = StandardScaler()
                axis   = scaler.fit_transform(xys)
                reg    = LinearRegression().fit(axis.reshape(-1, 2), np.array(colors).reshape(-1, 1))
                score  = reg.score(axis.reshape(-1, 2), np.array(colors).reshape(-1, 1))
                print("\t\t\t\t\t\t score", score)
                f_stats, p_values = f_regression(axis.reshape(-1, 2), np.array(colors))
                print("\t\t\t\t\t\t", "coefs", reg.coef_, reg.intercept_,
                      "stats", f_stats, "pvalue", p_values, "score", score)

                lap = layer_attribute_param
                lap[element][attribute].setdefault("coefs_x",    [])
                lap[element][attribute].setdefault("coefs_y",    [])
                lap[element][attribute].setdefault("intercept",  [])
                lap[element][attribute].setdefault("score",      [])
                lap[element][attribute].setdefault("fstats_f",   [])
                lap[element][attribute].setdefault("fstats_p",   [])
                lap[element][attribute].setdefault("square",     [])
                lap[element][attribute].setdefault("square_adj", [])
                lap[element][attribute].setdefault("pvalx",      [])
                lap[element][attribute].setdefault("pvaly",      [])

                mod = sm.OLS(colors, axis.reshape(-1, 2), fit_intercept=True)
                fii = mod.fit()
                p_vals = fii.summary2().tables[1]["P>|t|"]
                lap[element][attribute]["coefs_x"].append(reg.coef_[0, 0])
                lap[element][attribute]["coefs_y"].append(reg.coef_[0, 1])
                lap[element][attribute]["intercept"].append(reg.intercept_[0])
                lap[element][attribute]["score"].append(score)
                lap[element][attribute]["fstats_f"].append(fii.fvalue)
                lap[element][attribute]["fstats_p"].append(fii.f_pvalue)
                lap[element][attribute]["square"].append(fii.rsquared)
                lap[element][attribute]["square_adj"].append(fii.rsquared_adj)
                lap[element][attribute]["pvalx"].append(p_vals["x1"])
                lap[element][attribute]["pvaly"].append(p_vals["x2"])
                print("\t\t\t\t\t\t", layer_attribute_param)

                lower = pca_lower_dim(flattened, n_components=2)
                is_dimensional = attribute in (Attribute.VALENCE, Attribute.AROUSAL)
                new_colors     = [
                    colors[d] if is_dimensional else str(colors[d])
                    for d in range(len(data))
                ]
                new_lower      = np.asarray(lower)

                plot_embedding(
                    new_lower,
                    title=(f"embeddings {data_name} {model_name} {element} "
                           f"{attribute.name} ;sil_h {sil_h:.2f}"),
                    colors=new_colors,
                )

        model.history.clear()
        gc.collect()


print("DONE:\n", layer_attribute_param)
