import os.path

from ResidualMaskingNetwork.rmn import RMN
from data_loader.fer2013_data_loader import FER2013DataSet, FER2013DataLoader, Emotions

from config import configuration
from objects.context import get_context
from train.train import Train
from utils.filesystem.fs import FS

from models.net_smaller import Net as SmallerNet
from models.net import Net
from utils.init_system import init


def get7labels():
    return [Emotions.HAPPINESS.value, Emotions.SADNESS.value, Emotions.NEUTRAL.value, Emotions.ANGER.value,
     Emotions.FEAR.value, Emotions.DISGUST.value, Emotions.SURPRISE.value]

def get3labels():
    return [Emotions.HAPPINESS.value, Emotions.SADNESS.value, Emotions.NEUTRAL.value]

def train(batch_size, save_path, task_name, restore_path, labels_size=3, model_type=Net, alpha=1, p_layer=None, iden_emo=None, remove_iden=False):
    assert p_layer is None or p_layer in [1,3], f"p_layer must be None, 1 or 3, but got {p_layer}"
    assert iden_emo is None or iden_emo in ["emo", "iden"], f"iden_emo must be None, 'emo' or 'iden', but got {iden_emo}"
    assert labels_size in [3, 7], f"labels_size must be 3 or 7, but got {labels_size}"
    assert model_type in [Net, SmallerNet], f"model_type must be Net or SmallerNet, but got {model_type}"

    labels = get3labels()
    if labels_size == 7:
        labels = get7labels()

    fer13 = FER2013DataLoader()

    train, val, test = fer13.get_train_val_test(batch_size, labels=labels)
    train_run = Train(
        get_context(),
        train=train,
        val=val,
        test=test,
        iter_limit=1_000_000,
        n_classes=len(labels),
        epochs=100,
        log=50,
        save_path=os.path.join(save_path, task_name),
        restore_path=restore_path,
        net_type=model_type,
        alpha=alpha,
        p_layer=p_layer,
        iden_emo=iden_emo,
        remove_iden=remove_iden,
    )
    train_metrics, val_metrics, test_metrics, saved_path = train_run.run()


def eval(test=None, path=None, model=None, labels_size=3, net_type=None, remove_iden=None):
    assert labels_size in [3, 7], f"labels_size must be 3 or 7, but got {labels_size}"

    if test is None:
        fer13 = FER2013DataLoader()
        labels = get3labels()
        if labels_size == 7:
            labels = get7labels()
        train, val, test = fer13.get_train_val_test(1, labels=labels)


    train_run = Train(
        get_context(),
        test=test,
        iter_limit=1_000_000,
        n_classes=len(labels),
        epochs=100,
        log=50,
        restore_path=path,
        net_type=net_type,
        model=model,
        remove_iden=remove_iden
    )
    metrics = train_run.test()
    print(metrics)



for task_name, label_size, p, alpha_kmean, iden_emo, remove_iden, model in [
    ("em3_normal",3, None, 0, None, False, Net),
    ("em7_normal",7, None, 0, None, False, Net),
    ("em3_smaller",3, None, 0, None, False, SmallerNet),
    ("em3_no_p1_iden_-0.25", 3, 1, -0.25, "iden", False, Net),
    ("em3_yes_p3_iden", 3, 3, 1, "iden", False, Net),
    ("em3_no_raw_iden", 3, None, 1, None, True, Net),
    ("em3_no_p3_dimensional", 3, 3, 1, "emo", False, Net),
]:
    print(f"---------------{task_name}---------------------")
    train(
        configuration.data.batch_size, configuration.model.save_path, task_name, configuration.model.restore_path,
        p_layer=p, labels_size=label_size, model_type=model, alpha=alpha_kmean, iden_emo=iden_emo, remove_iden=remove_iden,
    )
    eval(path=f"{os.path.join(configuration.model.save_path, task_name, 'epoch99_last.pth')}",
         labels_size=label_size, net_type=model, remove_iden=remove_iden)

