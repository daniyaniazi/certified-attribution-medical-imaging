"""Model factory for ResNet, DenseNet, EfficientNet and MobileNet.

Enhancements:
- Adds optional extra fully-connected layers, BatchNorm1d and Dropout
  in the classifier head to improve classification performance.
- Adds MobileNet-v2 support.
- Keeps ImageNet pretrained weights by default (pretrained=True).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ModelConfig:
    """Configuration for a model."""
    def __init__(
        self,
        input_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        target_layer: str = 'layer4',
        backbone: str = 'resnet18',
        pretrained: bool = True
    ):
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.target_layer = target_layer
        self.backbone = backbone
        self.pretrained = pretrained


def _make_classifier(
    in_features: int,
    num_classes: int,
    extra_fc_layers: int = 0,
    batchnorm: bool = False,
    dropout: float = 0.0,
    hidden_dim: int = None
) -> nn.Module:
    """Build a flexible classifier head.

    Args:
        in_features: input feature size from backbone
        num_classes: number of output classes
        extra_fc_layers: number of extra hidden linear layers before final
        batchnorm: whether to add BatchNorm1d after each hidden layer
        dropout: dropout probability after activation
        hidden_dim: hidden unit size (defaults to in_features // 2)
    """
    if extra_fc_layers <= 0:
        if dropout > 0.0:
            return nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
        return nn.Linear(in_features, num_classes)

    if hidden_dim is None:
        hidden_dim = max(in_features // 2, 32)

    layers = []
    prev = in_features
    for i in range(extra_fc_layers):
        layers.append(nn.Linear(prev, hidden_dim))
        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        prev = hidden_dim

    layers.append(nn.Linear(prev, num_classes))
    return nn.Sequential(*layers)


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = 'cpu',
    extra_fc_layers: int = 0,
    batchnorm: bool = False,
    dropout: float = 0.0,
    hidden_dim: int = None
) -> Tuple[nn.Module, ModelConfig]:
    """Get model and its configuration with flexible classifier head.

    Args:
        model_name: 'resnet18', 'resnet50', 'densenet121', 'efficientnet_b0', 'mobilenet_v2', etc.
        num_classes: number of output classes
        pretrained: whether to use ImageNet pretrained weights (default True)
        device: 'cpu' or 'cuda'
        extra_fc_layers: number of extra hidden FC layers in the head
        batchnorm: add BatchNorm1d after hidden layers
        dropout: dropout probability in head
        hidden_dim: hidden dimension for extra FC layers

    Returns:
        (model, config)
    """
    model_name = model_name.lower()

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        model.fc = classifier
        config = ModelConfig(target_layer='layer4', backbone='resnet18', pretrained=pretrained)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        model.fc = classifier
        config = ModelConfig(target_layer='layer4', backbone='resnet50', pretrained=pretrained)

    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=pretrained)
        in_features = model.classifier.in_features
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        # DenseNet expects classifier to be a single Linear; wrap if Sequential
        if isinstance(classifier, nn.Sequential):
            model.classifier = classifier
        else:
            model.classifier = classifier
        config = ModelConfig(target_layer='features', backbone='densenet121', pretrained=pretrained)

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        # EfficientNet classifier is usually [Dropout, Linear]
        in_features = model.classifier[1].in_features
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        if isinstance(classifier, nn.Sequential):
            # Prepend dropout if specified and the Sequential doesn't already contain it
            model.classifier = classifier
        else:
            model.classifier[1] = classifier
        config = ModelConfig(target_layer='features', backbone='efficientnet_b0', pretrained=pretrained)

    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        if isinstance(classifier, nn.Sequential):
            model.classifier = classifier
        else:
            model.classifier[1] = classifier
        config = ModelConfig(target_layer='features', backbone='efficientnet_b1', pretrained=pretrained)

    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        # MobileNetV2 classifier is typically [Dropout, Linear]
        # Find in_features from last linear layer
        try:
            in_features = model.classifier[1].in_features
        except Exception:
            # Fallback
            in_features = model.last_channel if hasattr(model, 'last_channel') else 1280
        classifier = _make_classifier(in_features, num_classes, extra_fc_layers, batchnorm, dropout, hidden_dim)
        if isinstance(classifier, nn.Sequential):
            model.classifier = classifier
        else:
            model.classifier[1] = classifier
        config = ModelConfig(target_layer='features', backbone='mobilenet_v2', pretrained=pretrained)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    return model, config


def freeze_backbone(model: nn.Module, freeze_until_layer: str = None):
    """Freeze backbone parameters for fine-tuning.

    If `freeze_until_layer` is None, freezes all parameters except classifier/fc.
    Otherwise, unfreezes parameters starting at the matched layer name.
    """
    if freeze_until_layer is None:
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name and 'head' not in name:
                param.requires_grad = False
    else:
        freeze = True
        for name, param in model.named_parameters():
            if freeze_until_layer in name:
                freeze = False
            param.requires_grad = not freeze


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
