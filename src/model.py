"""
Backbone factory. Sticks to torchvision so there's no extra dependency (timm optional
upgrade later if you want more backbone choices).

All backbones are adapted to output a single logit (binary classification:
sigmoid(logit) = P(FAKE)).
"""
import torch
import torch.nn as nn
from torchvision import models


def get_model(backbone: str = "efficientnet_b0", pretrained: bool = True) -> nn.Module:
    backbone = backbone.lower()

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        in_features = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_features, 1)
        return net

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 1)
        return net

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        in_features = net.fc.in_features
        net.fc = nn.Linear(in_features, 1)
        return net

    raise ValueError(f"Unknown backbone '{backbone}'. Choose from: efficientnet_b0, resnet50, resnet18")


def get_last_conv_layer(model: nn.Module, backbone: str):
    """Returns the layer to hook for Grad-CAM, per backbone."""
    backbone = backbone.lower()
    if backbone == "efficientnet_b0":
        return model.features[-1]
    if backbone in ("resnet50", "resnet18"):
        return model.layer4[-1]
    raise ValueError(f"No known Grad-CAM target layer for backbone '{backbone}'")
