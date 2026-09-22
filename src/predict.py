"""Single-image inference pipeline, shared by app.py and the error-analysis notebook."""
import json
import os

import cv2
import numpy as np
import torch

from src.model import load_model
from src.preprocessing import preprocess_for_inference, IMG_SIZE
from src.dataset import LABEL_TO_NAME
from src.gradcam import GradCAM, overlay_heatmap

MODEL_PATH = "models/best_model.pt"
THRESHOLD_PATH = "results/metrics/uncertainty_threshold.json"


def _load_uncertainty_threshold():
    if os.path.exists(THRESHOLD_PATH):
        with open(THRESHOLD_PATH) as f:
            return json.load(f)["confidence_threshold"]
    return None  # not computed yet -- UNCERTAIN band stays off until evaluate.py has run


class Predictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(model_path, self.device)
        self.uncertainty_threshold = _load_uncertainty_threshold()

    def predict(self, image_bgr: np.ndarray):
        """Returns a dict with label, confidence, and the cropped face (or an error if no face found)."""
        face_rgb, input_tensor = preprocess_for_inference(image_bgr)
        if face_rgb is None:
            return {"error": "No face detected in the uploaded image."}

        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            prob_manipulated = torch.sigmoid(self.model(input_tensor)).item()

        pred_label = 1 if prob_manipulated >= 0.5 else 0
        confidence = prob_manipulated if pred_label == 1 else 1 - prob_manipulated

        label_name = LABEL_TO_NAME[pred_label]
        if self.uncertainty_threshold is not None and confidence < self.uncertainty_threshold:
            label_name = "UNCERTAIN"

        return {
            "label": label_name,
            "confidence": confidence,
            "prob_manipulated": prob_manipulated,
            "face_rgb": face_rgb,
            "input_tensor": input_tensor,
        }

    def predict_with_gradcam(self, image_bgr: np.ndarray):
        result = self.predict(image_bgr)
        if "error" in result:
            return result
        target_layer = self.model.features[-1]
        cam = GradCAM(self.model, target_layer)
        heatmap = cam.generate(result["input_tensor"])
        # heatmap matches the model's input size (IMG_SIZE), not the larger stored face crop
        face_resized = cv2.resize(result["face_rgb"], (IMG_SIZE, IMG_SIZE))
        result["gradcam_overlay"] = overlay_heatmap(face_resized, heatmap)
        return result


if __name__ == "__main__":
    import sys
    predictor = Predictor()
    image = cv2.imread(sys.argv[1])
    result = predictor.predict(image)
    print({k: v for k, v in result.items() if k not in ("face_rgb", "input_tensor")})
