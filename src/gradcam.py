"""Manual Grad-CAM (no extra dependency -- just forward/backward hooks).

Grad-CAM shows which pixels pushed the model toward its prediction. It is a
visual aid for understanding the model, not evidence that those pixels are
literally manipulated -- the heatmap should always be read as "the model
looked here", never as "this region is proven fake".
"""
import cv2
import numpy as np
import torch


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """input_tensor: (1, 3, H, W). Returns a (H, W) heatmap normalized to [0, 1]."""
        self.model.zero_grad()
        logit = self.model(input_tensor)
        logit.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # global-average-pool gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        return cv2.resize(cam, (target_w, target_h))


def overlay_heatmap(face_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (face_rgb * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    return overlay
