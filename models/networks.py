# -*- coding: utf-8 -*-
"""模型定义（Step 3 / Step 4）。

支持：
    resnet18          ImageNet 预训练基线
    resnet18_cbam     在 layer3 / layer4 末端嵌入 CBAM（通道注意力 + 空间注意力），
                      预训练主干照常加载，只有 CBAM 模块是新初始化的
    convnext_tiny     torchvision 的 ConvNeXt-Tiny（Transformer 时代的卷积网络，做架构对比）

small_stem 开关（仅 resnet18 系列）：
    标准 ResNet 首层:   conv 7x7 / stride 2  ->  maxpool 3x3 / stride 2
    小分辨率单字数据:   两次下采样会把 128x128 直接压到 32x32，一个 64x64 的汉字
                        放大到 128 后只剩 32 像素，笔锋细节基本被抹平。
    small_stem=True:    conv 3x3 / stride 1  ->  去掉 maxpool
                        送入 layer1 的特征图仍是 128x128，保留细笔画；代价是显存/算力上升。

预训练权重：默认加载 ImageNet 权重。若网络不可用会降级为随机初始化并给出警告。
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights


# --------------------------------------------------------------------------------------
# CBAM
# --------------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    """通道注意力：avg/max 两路池化过同一个共享 MLP，相加后 sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * att


class SpatialAttention(nn.Module):
    """空间注意力：沿通道维做 avg/max 拼接成 2 通道，7x7 卷积后 sigmoid。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.max(x, dim=1, keepdim=True).values
        att = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class CBAM(nn.Module):
    """标准 CBAM：先通道注意力，再空间注意力（串行）。"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class ResNet18CBAM(nn.Module):
    """resnet18 + 在 layer3 / layer4 末端插入 CBAM。"""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"加载 ImageNet 预训练失败，改为随机初始化: {type(exc).__name__}: {exc}")
            net = torchvision.models.resnet18(weights=None)

        # 预训练权重已在构造时载入，这里只在其后追加新的注意力模块
        net.layer3 = nn.Sequential(net.layer3, CBAM(256))
        net.layer4 = nn.Sequential(net.layer4, CBAM(512))
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        if dropout > 0:
            net.fc = nn.Sequential(nn.Dropout(dropout), net.fc)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _center_crop_7to3(w: torch.Tensor) -> torch.Tensor:
    """把预训练 conv1 的 7x7 卷积核中心裁成 3x3，作为 small_stem 的初始化。

    直接用随机初始化会丢掉 ImageNet 学到的边缘/方向滤波器，小数据集上更吃亏。
    """
    return w[:, :, 2:5, 2:5].clone()


class SmallStemResNet18(nn.Module):
    """resnet18 + 小步长 stem（3x3/s1，无 maxpool）。"""

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"加载 ImageNet 预训练失败，改为随机初始化: {type(exc).__name__}: {exc}")
            net = torchvision.models.resnet18(weights=None)

        old_conv = net.conv1
        new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if old_conv.weight.shape[-1] == 7:
            with torch.no_grad():
                new_conv.weight.copy_(_center_crop_7to3(old_conv.weight))
        net.conv1 = new_conv
        net.maxpool = nn.Identity()          # 保留分辨率
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        if dropout > 0:
            net.fc = nn.Sequential(nn.Dropout(dropout), net.fc)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------------------
# 构建入口
# --------------------------------------------------------------------------------------
MODEL_NAMES = ["resnet18", "resnet18_cbam", "convnext_tiny"]


def build_model(
    name: str = "resnet18",
    num_classes: int = 20,
    small_stem: bool = False,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """构建模型。

    name: resnet18 / resnet18_cbam / convnext_tiny
    small_stem: 仅对 resnet18 系列有效（True 时用 3x3/s1 + 去掉 maxpool 的 stem）
    """
    name = name.lower()
    if name not in MODEL_NAMES:
        raise ValueError(f"未知模型 {name}，可选: {MODEL_NAMES}")

    if name == "resnet18":
        if small_stem:
            return SmallStemResNet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = torchvision.models.resnet18(weights=weights)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"加载 ImageNet 预训练失败，改为随机初始化: {type(exc).__name__}: {exc}")
            net = torchvision.models.resnet18(weights=None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        if dropout > 0:
            net.fc = nn.Sequential(nn.Dropout(dropout), net.fc)
        return net

    if name == "resnet18_cbam":
        if small_stem:
            warnings.warn("resnet18_cbam 暂不支持 small_stem，已忽略该开关")
        return ResNet18CBAM(num_classes=num_classes, pretrained=pretrained, dropout=dropout)

    if name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = torchvision.models.convnext_tiny(weights=weights)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"加载 ConvNeXt 预训练失败，改为随机初始化: {type(exc).__name__}: {exc}")
            net = torchvision.models.convnext_tiny(weights=None)
        in_dim = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_dim, num_classes)
        return net

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def describe(model: nn.Module) -> str:
    total = count_parameters(model)
    trainable = count_parameters(model, True)
    return f"{model.__class__.__name__}: {total / 1e6:.2f}M 参数（可训练 {trainable / 1e6:.2f}M）"


if __name__ == "__main__":
    for nm in MODEL_NAMES:
        m = build_model(nm, 20, pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            y = m(x)
        print(f"{nm:<14s} out={tuple(y.shape)}  {describe(m)}")


