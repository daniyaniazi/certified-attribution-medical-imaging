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
    """Multi-head grid model with per-cell disconnection (DiFull-style).

    Forward crops the requested cell before running the shared backbone so each head
    can only depend on its own cell. When head_id is None, logits for all cells are
    returned, each computed from its corresponding crop.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 8,
        num_heads: int = 4,
        pretrained: bool = True,
        scale: int = 2,
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
        self.scale = scale

    def _crop_cell(self, x: torch.Tensor, cell_id: int) -> torch.Tensor:
        """Extract a single cell from the full grid tensor."""
        b, c, full_h, full_w = x.shape
        cell_h = full_h // self.scale
        cell_w = full_w // self.scale
        row = cell_id // self.scale
        col = cell_id % self.scale
        y0, y1 = row * cell_h, (row + 1) * cell_h
        x0, x1 = col * cell_w, (col + 1) * cell_w
        return x[:, :, y0:y1, x0:x1]

    def forward(self, x: torch.Tensor, head_id: Optional[int] = None):
        if head_id is not None:
            cell = self._crop_cell(x, head_id)
            feats = self.feature_extractor(cell)
            return self.heads[head_id](feats)  # [B, C]

        # All heads
        logits = []
        for i, head in enumerate(self.heads):
            cell = self._crop_cell(x, i)
            feats = self.feature_extractor(cell)
            logits.append(head(feats))
        return torch.stack(logits, dim=1)  # [B, K, C]

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
