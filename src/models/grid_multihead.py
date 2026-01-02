"""Multi-head grid model wrapper.

Wraps a backbone (e.g., resnet18) and replicates the classifier head K times (K = num_heads).
Forward can return logits for all heads or a single head via head_id.
"""
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.body = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.out_features = backbone.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        return torch.flatten(feats, 1)


class GridMultiHead(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 8,
        num_heads: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        backbone_name = backbone_name.lower()
        if backbone_name == "resnet18":
            base = models.resnet18(pretrained=pretrained)
            self.feature_extractor = ResNetFeatureExtractor(base)
        else:
            raise ValueError(f"Unsupported backbone for grid: {backbone_name}")

        self.heads = nn.ModuleList([
            nn.Linear(self.feature_extractor.out_features, num_classes)
            for _ in range(num_heads)
        ])
        self.num_heads = num_heads
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, head_id: Optional[int] = None):
        feats = self.feature_extractor(x)
        if head_id is None:
            return torch.stack([head(feats) for head in self.heads], dim=1)  # [B, K, C]
        return self.heads[head_id](feats)  # [B, C]

    def freeze_backbone(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def duplicate_head_weights(self, source_head: int = 0):
        """Copy weights from source_head to all heads (useful when loading single-head checkpoints)."""
        with torch.no_grad():
            src_w = self.heads[source_head].weight.data.clone()
            src_b = self.heads[source_head].bias.data.clone()
            for h_idx, head in enumerate(self.heads):
                if h_idx == source_head:
                    continue
                head.weight.data.copy_(src_w)
                head.bias.data.copy_(src_b)
