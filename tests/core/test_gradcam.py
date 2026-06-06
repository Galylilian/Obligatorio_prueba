import torch
import numpy as np
from src.core.model import get_model
from src.explainability.gradcam import GradCAM


def test_gradcam_output():
    """
    Verifica que Grad-CAM genera un mapa válido.
    """

    model = get_model()
    model.eval()

    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    dummy_input = torch.randn(1, 3, 224, 224)

    cam = grad_cam.generate(dummy_input)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
