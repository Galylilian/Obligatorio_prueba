import torch.nn as nn
import torchvision.models as models


def get_model(num_classes: int = 2) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_from_layer4(model: nn.Module) -> None:
    for param in model.layer4.parameters():
        param.requires_grad = True


def trainable_parameter_groups(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
