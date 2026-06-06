#Se utilizó un modelo ResNet18 preentrenado sobre el cual se aplicó fine-tuning, entrenando todas las capas del modelo para adaptarlo al problema específico.
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model