"""MobileNetV2 transfer-learning model for REAL vs MANIPULATED classification.

MobileNetV2 was chosen over ResNet/EfficientNet for this project because it is
~14MB, trains noticeably faster on CPU, and its accuracy on face-realism tasks
is close enough to bigger backbones that the tradeoff favors a laptop workflow.
That tradeoff (a few points of possible accuracy vs. practical training time)
is exactly the kind of decision an interviewer will ask about.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def build_model(pretrained: bool = True) -> nn.Module:
    weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v2(weights=weights)
    # Single output logit (not 2-class softmax): with BCEWithLogitsLoss, sigmoid(logit)
    # is directly P(manipulated), which is what the confidence score in the UI needs.
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model


def save_checkpoint(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, device: torch.device) -> nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
