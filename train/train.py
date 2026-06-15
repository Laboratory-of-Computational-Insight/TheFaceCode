import gc
import os
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch

import data_load
from data_load import disfa_train

from models.net import Net
from objects.context import Context
from utils.model import create_model


# ─────────────────────────────────────────────────────────────────────────────
# Identity-loss configuration
# ─────────────────────────────────────────────────────────────────────────────

class IdentityLossMode:
    """How the auxiliary identity k-means loss is combined with the CE loss."""
    NONE           = "none"           # baseline — no identity loss at all
    ALTERNATING    = "alternating"    # original: separate DISFA backward after each FER batch
    MIXED          = "mixed"          # reviewer-recommended: single backward with combined loss
    EMOTION_KMEANS = "emotion_kmeans" # condition (4): k-means on emotion labels at P3, FER2013 only


@dataclass
class IdentityLossConfig:
    """
    All parameters that control the auxiliary identity loss in Study 5.

    mode:
        NONE        — standard emotion training, no identity term
        ALTERNATING — legacy mode: N separate DISFA steps per FER batch
        MIXED       — single forward/backward with CE + alpha*kmeans

    alpha:
        Multiplier on the identity k-means loss.
        Positive  (+) → encourages identity clustering  (retain-identity condition)
        Negative  (-) → discourages identity clustering (suppress-identity condition)

    layer:
        1 → apply loss at P1 (model.inner_p1)
        3 → apply loss at P3 (model.inter_p3)

    disfa_steps_per_fer_batch:
        Only used in ALTERNATING mode.
        How many separate DISFA backward passes per FER batch.
        Original paper used 50 for P1 and 1 for P3.

    pca_dims:
        PCA reduction applied to activations before computing k-means loss.
        Matches original training (4 dims for P1, 2 dims for P3).

    disfa_batch_size:
        Samples drawn from DISFA per training step.
    """
    mode:    str   = IdentityLossMode.NONE
    alpha:   float = 1.0
    layer:   int   = 3
    disfa_steps_per_fer_batch: int = 1
    pca_dims:                  int = 2
    disfa_batch_size:          int = 270

    @classmethod
    def retain_identity_p3_alternating(cls) -> "IdentityLossConfig":
        """Original Study-5 result (3): retain identity at P3, alternating training."""
        return cls(mode=IdentityLossMode.ALTERNATING, alpha=2.0, layer=3,
                   disfa_steps_per_fer_batch=1, pca_dims=2)

    @classmethod
    def retain_identity_p3_mixed(cls) -> "IdentityLossConfig":
        """Reviewer C3.4 suggestion: same objective but with mixed minibatches."""
        return cls(mode=IdentityLossMode.MIXED, alpha=2.0, layer=3, pca_dims=2)

    @classmethod
    def suppress_identity_p1_alternating(cls) -> "IdentityLossConfig":
        """Original Study-5 result (1): suppress identity at P1, alternating training."""
        return cls(mode=IdentityLossMode.ALTERNATING, alpha=-0.25, layer=1,
                   disfa_steps_per_fer_batch=50, pca_dims=1_000)  # random 1k features

    @classmethod
    def suppress_identity_p1_mixed(cls) -> "IdentityLossConfig":
        """Mixed-minibatch version of result (1)."""
        return cls(mode=IdentityLossMode.MIXED, alpha=-0.25, layer=1, pca_dims=1_000)

    @classmethod
    def force_emotion_clustering_p3(cls) -> "IdentityLossConfig":
        """Study-5 result (4): force emotion clustering at P3 using FER2013 emotion labels.
        No DISFA data needed — uses the emotion class labels already in each FER2013 batch."""
        return cls(mode=IdentityLossMode.EMOTION_KMEANS, alpha=1.0, layer=3, pca_dims=2)


def randomize_disfa(batch_size=270):
    if disfa_load.data is not None and disfa_load.labels is not None:
        return disfa_load.data, disfa_load.labels

    disfa_load.DISFA = disfa_train()
    data   = []
    labels = []
    keys   = list(disfa_load.DISFA.keys())
    np.random.shuffle(keys)
    keys = keys[:int(0.8 * len(keys))]
    for key in keys:
        np.random.shuffle(disfa_load.DISFA[key])
        data.extend(disfa_load.DISFA[key][:batch_size // len(keys)])
        labels.extend([key] * (batch_size // len(keys)))
    disfa_load.data   = torch.stack(data)
    disfa_load.labels = torch.tensor(labels)
    return disfa_load.data, disfa_load.labels


class Train:
    def __init__(
        self,
        context: Context,
        n_classes=7,
        iter_limit=2500,
        epochs=12,
        last_epoch=-1,
        restore_path=None,
        save_path=None,
        optimizer_lr=1e-3,
        optimizer_beta=(0.9, 0.999),
        optimzier_eps=1e-07,
        schedualer_step=2,
        schedualer_gamma=0.2,
        train=None,
        val=None,
        test=None,
        log=0,
        model=None,
        net_type=Net,
        identity_loss: Optional[IdentityLossConfig] = None,
        pixel_pca_k: int = 0,
        pixel_pca_components: Optional[np.ndarray] = None,
        pixel_pca_mean: Optional[np.ndarray] = None,
        # Legacy params kept for backward compat (used by em3_no_raw_iden)
        remove_iden=None,
    ):
        self.nclasses   = n_classes
        self.context    = context
        self.iter_limit = iter_limit
        self.epochs     = epochs
        self.last_epoch = last_epoch
        self.restore_path = restore_path
        self.save_path    = save_path

        self.model, self.last_epoch = model, -1
        if model is None:
            self.model, self.last_epoch = create_model(
                context, n_classes, net_type, restore_path, last_epoch=last_epoch
            )
        if remove_iden is not None:
            self.model.toggle_remove_iden(remove_iden)

        self.optimizer = optim.NAdam(
            self.model.parameters(),
            lr=optimizer_lr, betas=optimizer_beta, eps=optimzier_eps,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=7, min_lr=1e-7
        )
        self.loss      = CrossEntropyLoss()
        self.train_dl  = train
        self.val_dl    = val
        self.test_dl   = test
        self.log       = log
        self.identity_loss = identity_loss or IdentityLossConfig()  # default = NONE

        # ── pixel-space PCA removal ───────────────────────────────────────────
        self.pixel_pca_k = pixel_pca_k
        if pixel_pca_k > 0 and pixel_pca_components is not None:
            self.pca_comp = torch.from_numpy(pixel_pca_components[:pixel_pca_k]).float()
            self.pca_mean = torch.from_numpy(pixel_pca_mean).float()
        else:
            self.pca_comp = None
            self.pca_mean = None

    def reset_model_params(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def kmean_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        output: (N, D) embeddings
        labels: (N,) integer cluster labels
        """
        unique_labels = torch.unique(labels)
        stds = output.new_tensor(0.0)
        for label in unique_labels:
            label_rows = output[labels == label]
            stds += label_rows.std()
        return stds / len(unique_labels)

    def _get_identity_activation(self, model, layer: int, pca_dims: int):
        """Extract and PCA-reduce activations at P1 or P3."""
        if layer == 1:
            inner = model.inner_p1
            inner = inner.reshape(len(inner), -1)
            if pca_dims >= inner.shape[1]:
                return inner
            indexes = torch.randperm(inner.shape[1])[:pca_dims]
            return inner[:, indexes]
        else:  # P3
            inner = model.inter_p3
            inner = inner.reshape(len(inner), -1)
            U, S, V = torch.pca_lowrank(inner)
            return (inner - inner.mean(dim=0)) @ V[:, :pca_dims]

    def _remove_pixel_pca(self, images: torch.Tensor) -> torch.Tensor:
        """Project out top-k pixel PCA components from a batch of images."""
        shape = images.shape
        flat  = images.reshape(len(images), -1) - self.pca_mean
        proj  = flat @ self.pca_comp.T @ self.pca_comp
        return (flat - proj + self.pca_mean).reshape(shape)

    def iterate_batch(self, model, optimizer, loss, data, train, log: bool = False):
        cfg    = self.identity_loss
        images = data[0].float()
        labels = data[1]

        if self.pca_comp is not None:
            images = self._remove_pixel_pca(images)

        with ExitStack() as stack:
            if not train:
                stack.enter_context(torch.no_grad())

            if train:
                optimizer.zero_grad()

            # ── forward on FER2013 batch ──────────────────────────────────────
            preds  = model(images.to(self.context.device))
            labels = labels.to(self.context.device)
            l1     = loss(preds, labels)

            # ── combined loss ─────────────────────────────────────────────────
            l_combined = l1

            # Condition (4): emotion k-means at P3 — uses FER2013 labels, no DISFA needed
            if train and cfg.mode == IdentityLossMode.EMOTION_KMEANS:
                inner      = self._get_identity_activation(model, 3, cfg.pca_dims)
                l_combined = l1 + cfg.alpha * self.kmean_loss(inner, labels)

            if train and cfg.mode == IdentityLossMode.MIXED:
                disfa_data, disfa_labels = randomize_disfa(cfg.disfa_batch_size)
                disfa_data   = disfa_data.to(self.context.device)
                disfa_labels = disfa_labels.to(self.context.device)
                model(disfa_data)
                inner      = self._get_identity_activation(model, cfg.layer, cfg.pca_dims)
                l_identity = cfg.alpha * self.kmean_loss(inner, disfa_labels)
                l_combined = l1 + l_identity

            if train:
                l_combined.backward()
                optimizer.step()
                optimizer.zero_grad()

            acc   = (preds.argmax(1) == labels).float().mean().to(self.context.cpu_device)
            l_log = l_combined.detach().clone().to(self.context.cpu_device)

            # ── alternating mode: separate DISFA steps after FER step ─────────
            if train and cfg.mode == IdentityLossMode.ALTERNATING:
                for _ in range(cfg.disfa_steps_per_fer_batch):
                    optimizer.zero_grad()
                    disfa_data, disfa_labels = randomize_disfa(cfg.disfa_batch_size)
                    disfa_data   = disfa_data.to(self.context.device)
                    disfa_labels = disfa_labels.to(self.context.device)
                    model(disfa_data)
                    inner = self._get_identity_activation(model, cfg.layer, cfg.pca_dims)
                    l_alt = cfg.alpha * self.kmean_loss(inner, disfa_labels)
                    l_alt.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        print({"acc": acc, "loss": l_log})
        return acc, l_log

    @staticmethod
    def metric_to_list(metrics):
        return [t.tolist() for t in metrics]

    @staticmethod
    def allocate_metrics(n):
        return [
            torch.zeros(n),  # accuracy
            torch.zeros(n),  # loss
        ]

    def iterate_epoch(self, iter_limit, model, optimizer, loss, data, train=True, log: int = 0):
        n_batch = min(iter_limit, len(data))
        metrics = self.allocate_metrics(n_batch)

        for batch_idx, batch_data in enumerate(data):
            if batch_idx >= iter_limit:
                break
            log_iteration = (batch_idx % log) == log - 1 or batch_idx == n_batch - 1

            res = self.iterate_batch(model, optimizer, loss, batch_data, train, log=log_iteration)

            for i in range(len(res)):
                metrics[i][batch_idx] = res[i]

            if log is not None and log_iteration:
                print(
                    f"{batch_idx} metrics:",
                    [metrics[i][batch_idx].item() for i in range(len(metrics))],
                )

        means = [metrics[i].mean() for i in range(len(metrics))]
        return means

    def run(self):
        saved_path      = ""
        prev_train_state = self.model.training

        train_metrics = self.allocate_metrics(self.epochs - self.last_epoch)
        val_metrics   = self.allocate_metrics(self.epochs - self.last_epoch)

        best = None
        for epoch_idx in tqdm(range(self.last_epoch + 1, self.epochs)):
            torch.cuda.empty_cache()
            self.model.zero_grad()
            gc.collect()

            self.model.train()
            print("----train----")
            train_res = self.iterate_epoch(
                self.iter_limit, self.model, self.optimizer, self.loss,
                self.train_dl, train=True, log=self.log,
            )
            for i in range(len(train_res)):
                train_metrics[i][epoch_idx - self.last_epoch] = train_res[i]

            with torch.no_grad():
                print("----eval----")
                self.model.eval()
                val_res = self.iterate_epoch(
                    self.iter_limit, self.model, self.optimizer, self.loss,
                    self.val_dl, train=False, log=self.log,
                )
                self.model.train(prev_train_state)

            for i in range(len(val_res)):
                val_metrics[i][epoch_idx - self.last_epoch] = val_res[i]
            print({"epoch acc": val_metrics[0][epoch_idx - self.last_epoch]})

            self.scheduler.step(val_metrics[0][epoch_idx - self.last_epoch])

            if epoch_idx % 10 == 0 or epoch_idx == self.epochs - 1:
                saved_path, best = self.save_model(epoch_idx, train_metrics, val_metrics, best)

        test_metrics = self.test()
        print("test metrics:", test_metrics)

        return train_metrics, val_metrics, test_metrics, saved_path

    def save_model(self, epoch_idx, train_metrics, val_metrics, best=None):
        if self.save_path is None:
            return None, best

        saved_path = f"epoch{epoch_idx}_last.pth"
        os.makedirs(self.save_path, exist_ok=True)
        local_path = os.path.join(self.save_path, saved_path)

        torch.save(
            {
                "epoch": epoch_idx,
                "metrics": dict(
                    val=self.metric_to_list(val_metrics),
                    train=self.metric_to_list(train_metrics),
                ),
                "state_dict": self.model.state_dict(),
                "n_classes":  self.model.n_classes,
            },
            local_path,
        )
        print(f"  Model saved: {local_path}")
        return saved_path, best

    def test(self):
        test_metrics         = self.allocate_metrics(1)
        previous_train_state = self.model.training
        self.model.eval()

        with torch.no_grad():
            test_res = self.iterate_epoch(
                self.iter_limit, self.model, self.optimizer, self.loss,
                self.test_dl, train=False, log=self.log,
            )

        for i in range(len(test_res)):
            test_metrics[i][0] = test_res[i]

        self.model.train(previous_train_state)
        return test_metrics
