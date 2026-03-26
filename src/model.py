# src/model.py
import torch
import torch.nn as nn
import timm


class DRClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, dropout_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.SiLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def build_model(num_classes=5, device='cpu', freeze_backbone=False):
    model = DRClassifier(num_classes=num_classes, pretrained=False)
    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    return model.to(device)


def get_target_layer(model):
    """Return last conv layer for Grad-CAM."""
    return model.backbone.blocks[-1][-1].conv_pwl