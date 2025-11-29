import torch
import io

from utils.filesystem.fs import FS


def create_model(context, n_classes, model, restore_path=None, device=None, last_epoch=-1):
    if device is None:
        device = context.device

    if restore_path is not None:
        data = FS().get_data(restore_path)
        model_load = torch.load(io.BytesIO(data), map_location=device)
        retrieved_n_classes = model_load.get("n_classes")
        if retrieved_n_classes is None:
            print(f"no classes found, using input {n_classes}")
            retrieved_n_classes = n_classes
        elif retrieved_n_classes != n_classes:
            print(f"retried n classes {retrieved_n_classes} and given n_classes {n_classes} are different")
        model = model(retrieved_n_classes).to(device)
        if not hasattr(model, "n_classes"):
            model.n_classes = retrieved_n_classes

        model.load_state_dict(model_load["state_dict"])

        last_epoch = model_load["epoch"]
    else:
        model = model(n_classes).to(device)

    model = model.to(device)

    return model, last_epoch
