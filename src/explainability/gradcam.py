import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.zero_grad()

        if not input_tensor.requires_grad:
            input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)
        class_idx = output.argmax()

        output[0, class_idx].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks no capturaron activaciones/gradientes")

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + img

    return overlay.astype(np.uint8)
