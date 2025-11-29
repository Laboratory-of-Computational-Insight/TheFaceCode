import torchvision
from torch import nn
from torchvision.models import VGG16_Weights


class WrapVGG(nn.Module):
    def __init__(self, device="cpu"):
        super(WrapVGG, self).__init__()
        self.vgg = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.vgg.to("cuda")

        self.history = []

    def forward(self, x):
        history = {}

        history_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(224)),
        ])
        def first_hook_fn(module, input, output):
            pass
            history["first"] = history_transforms(output).clone().detach().cpu()
        first_hook = self.vgg.features[0].register_forward_hook(first_hook_fn)
        def second_hook_fn(module, input, output):
            pass
            history["second"] = output.clone().detach().cpu()
        second_hook = self.vgg.avgpool.register_forward_hook(second_hook_fn)
        def third_hook_fn(module, input, output):
            pass
            history["third"] = output.clone().detach().cpu()

        third_hook = self.vgg.classifier[3].register_forward_hook(third_hook_fn)
        def fourth_hook_fn(module, input, output):
            pass
            history["forth"] = output.clone().detach().cpu()

        fourth_hook = self.vgg.classifier[6].register_forward_hook(fourth_hook_fn)
        x = x.repeat(1,3,1,1)
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        out = self.vgg(train_transforms(x))
        self.history.append(history)

        first_hook.remove()
        second_hook.remove()
        third_hook.remove()
        fourth_hook.remove()

        return out