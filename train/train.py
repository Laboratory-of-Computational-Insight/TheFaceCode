import gc
import io
import os
from contextlib import ExitStack

import numpy as np
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch

import data_load
from data_load import disfa_train

from models.net import Net
from objects.context import Context
from utils.filesystem.fs import FS
from utils.model import create_model


def randomize_disfa(batch_size=270):
    if disfa_load.data is not None and disfa_load.labels is not None:
        return disfa_load.data, disfa_load.labels

    disfa_load.DISFA = disfa_train()
    data = []
    labels = []
    keys = list(disfa_load.DISFA.keys())
    np.random.shuffle(keys)
    keys = keys[:int(0.8*len(keys))]
    for key in keys:
        np.random.shuffle(disfa_load.DISFA[key])
        data.extend(disfa_load.DISFA[key][:batch_size // len(keys)])
        labels.extend([key]*(batch_size//len(keys)))
    disfa_load.data = torch.stack(data)
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
        optimzier_eps = 1e-07,
        schedualer_step=2,
        schedualer_gamma=0.2,
        train=None,
        val=None,
        test=None,
        log=0,
        model=None,
        net_type = Net,
        alpha=0,
        iden_emo=None,
        p_layer=None,
        remove_iden=None,
    ):
        self.nclasses = n_classes
        self.context = context
        self.iter_limit = iter_limit
        self.epochs = epochs
        self.last_epoch = last_epoch
        self.restore_path = restore_path
        self.save_path = save_path

        self.model, self.last_epoch = model, -1
        if model is None:
            self.model, self.last_epoch = create_model(
                context, n_classes, net_type, restore_path, last_epoch=last_epoch
            )
        if remove_iden is not None:
            self.model.toggle_remove_iden(remove_iden)

        self.optimizer = optim.NAdam(
            self.model.parameters(),
            lr=optimizer_lr, betas=optimizer_beta, eps=optimzier_eps
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, patience=7, min_lr=1e-7
        )
        self.loss = CrossEntropyLoss()
        self.train_dl = train
        self.val_dl = val
        self.test_dl = test
        self.log = log

        self.alpha = alpha
        self.iden_emo = iden_emo
        self.p_layer = p_layer


    def reset_model_params(self):
        for layer in self.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def kmeans_loss(self, output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        output: (N, D) embeddings
        labels: (N,) integer cluster labels
        """
        unique_labels = torch.unique(labels)
        loss = output.new_tensor(0.0)

        for label in unique_labels:
            mask = labels == label
            points = output[mask]  # (n_k, D)
            if points.size(0) <= 1:
                continue  # no variance if only 0/1 point in cluster
            center = points.mean(dim=0, keepdim=True)  # (1, D)
            # mean squared distance to centroid (k-means objective per cluster)
            loss += ((points - center) ** 2).sum(dim=1).mean()
        return loss / len(unique_labels)

    def iterate_batch(self, model, optimizer, loss, data, train, log: bool = False):
        alpha = self.alpha
        iden_emo = self.iden_emo
        p = self.p_layer

        images = data[0].float()
        labels = data[1]

        with ExitStack() as stack:
            if not train:
                stack.enter_context(torch.no_grad())

            if train:
                optimizer.zero_grad()

            preds = model(images.to(self.context.device))

            labels = labels.to(self.context.device)
            l1 = loss(preds, labels)
            l2 = 0
            if p == 3 and iden_emo == "emo":
                inner = model.inter_p3
                inner = inner.reshape(len(inner), -1)
                U, S, V = torch.pca_lowrank(inner)
                inner = (inner - inner.mean(dim=0)) @ V[:, : 4]
                l2 = self.kmeans_loss(inner, labels)

            l = l1 + alpha * l2

            if train:
                l.backward()
                optimizer.step()
                optimizer.zero_grad()


            acc = (preds.argmax(1) == labels).float().mean().to(self.context.cpu_device)
            l = l.detach().clone().to(self.context.cpu_device)

            if train and iden_emo=="iden":
                iters = 2 if p == 1 else 1
                for i in range(iters):
                    optimizer.zero_grad()

                    data, labels = randomize_disfa()
                    data = data.to(self.context.device)
                    labels = labels.to(self.context.device)
                    model(data)
                    if p == 1:
                        inner = model.inner_p1
                        inner = inner.reshape(len(inner), -1)
                        indexes = torch.randperm(inner.shape[1])[:1_000]
                        inner = inner[:, indexes]
                        l2 = alpha * self.kmeans_loss(inner.to(self.context.device), labels)
                    elif p == 3:
                        inner = model.inter_p3
                        inner = inner.reshape(len(inner), -1)
                        U, S, V = torch.pca_lowrank(inner)
                        inner = (inner - inner.mean(dim=0)) @ V[:, : 2]
                        l2 = alpha * self.kmeans_loss(inner, labels)

                    l2.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        print({"acc": acc, "loss": l})

        return acc, l

    @staticmethod
    def metric_to_list(metrics):
        return [t.tolist() for t in metrics]

    @staticmethod
    def allocate_metrics(n):
        return [
            torch.zeros(n),  # accuracy
            torch.zeros(n),  # loss
        ]

    def iterate_epoch(
        self, iter_limit, model, optimizer, loss, data, train=True, log: int = 0
    ):
        n_batch = min(iter_limit, len(data))
        metrics = self.allocate_metrics(n_batch)

        for batch_idx, batch_data in enumerate(data):
            if batch_idx >= iter_limit:
                break
            log_iteration = (batch_idx % log) == log - 1 or batch_idx == n_batch - 1

            res = self.iterate_batch(
                model, optimizer, loss, batch_data, train, log=log_iteration
            )

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
        saved_path = ""
        prev_train_state = self.model.training

        train_metrics = self.allocate_metrics(self.epochs - self.last_epoch)
        val_metrics = self.allocate_metrics(self.epochs - self.last_epoch)

        best = None
        for epoch_idx in tqdm(range(self.last_epoch + 1, self.epochs)):
            torch.cuda.empty_cache()
            self.model.zero_grad()
            gc.collect()

            self.model.train()
            print("----train----")
            train_res = self.iterate_epoch(
                self.iter_limit,
                self.model,
                self.optimizer,
                self.loss,
                self.train_dl,
                train=True,
                log=self.log,
            )

            for i in range(len(train_res)):
                train_metrics[i][epoch_idx-self.last_epoch] = train_res[i]

            with torch.no_grad():
                print("----eval----")
                self.model.eval()
                val_res = self.iterate_epoch(
                    self.iter_limit,
                    self.model,
                    self.optimizer,
                    self.loss,
                    self.val_dl,
                    train=False,
                    log=self.log,
                )
                self.model.train(prev_train_state)

            for i in range(len(val_res)):
                val_metrics[i][epoch_idx-self.last_epoch] = val_res[i]
            print({"epoch acc": val_metrics[0][epoch_idx-self.last_epoch]})

            self.scheduler.step(val_metrics[0][epoch_idx-self.last_epoch])

            if epoch_idx % 10 == 0 or epoch_idx == self.epochs - 1:
                saved_path, best = self.save_model(epoch_idx, train_metrics, val_metrics, best)

        test_metrics = self.test()
        print("test metrics:", test_metrics)

        return train_metrics, val_metrics, test_metrics, saved_path

    def save_model(self, epoch_idx, train_metrics, val_metrics, best=None):
        if self.save_path is None:
            return None

        saved_path = f"epoch{epoch_idx}_last.pth"

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": epoch_idx,
                "metrics": dict(
                    val=self.metric_to_list(val_metrics),
                    train=self.metric_to_list(train_metrics),
                ),
                "state_dict": self.model.state_dict(),
                "n_classes": self.model.n_classes
            },
            buffer,
        )
        buffer.seek(0)
        FS().upload_data(buffer, os.path.join(self.save_path, saved_path))

        return saved_path, best

    def test(self):
        test_metrics = self.allocate_metrics(1)
        previous_train_state = self.model.training
        self.model.eval()

        with torch.no_grad():
            test_res = self.iterate_epoch(
                self.iter_limit,
                self.model,
                self.optimizer,
                self.loss,
                self.test_dl,
                train=False,
                log=self.log,
            )

        for i in range(len(test_res)):
            test_metrics[i][0] = test_res[i]

        self.model.train(previous_train_state)

        return test_metrics
