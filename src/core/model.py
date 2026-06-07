#Se utilizó un modelo ResNet18 preentrenado sobre el cual se aplicó fine-tuning, entrenando todas las capas del modelo para adaptarlo al problema específico.

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_model(pretrained: bool = False):
    """
    pretrained=True  -> entrenamiento
    pretrained=False -> inferencia (API / video)
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None  # No descarga pesos

    model = resnet18(weights=weights)

    # Reemplazar la capa final para clasificación binaria
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
