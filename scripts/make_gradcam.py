# -*- coding: utf-8 -*-
"""为最佳模型生成 8 张 Grad-CAM 图（官方测试集样本）。

    python scripts/make_gradcam.py --exp exp2_domain_aligned --arch resnet18
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import build_dataloaders, denormalize  # noqa: E402


def target_layer(model, arch: str):
    if arch == "convnext_tiny":
        return model.features[-1][-1]
    if arch == "resnet18_cbam":
        return model.net.layer4[0][-1]
    return model.layer4[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=128)
    a = ap.parse_args()

    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    fontm = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in fontm else "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ck = torch.load(ROOT / "checkpoints" / a.exp / "best.pth", map_location="cpu", weights_only=False)
    classes = ck["classes"]
    dev = torch.device("cuda")
    model = build_model(a.arch, num_classes=len(classes), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    dl = build_dataloaders(img_size=a.img_size, batch_size=64, num_workers=0,
                           domain_align=True, domain_fill=0.88)
    # 每类挑一张正确分类的样本
    picked, per_class = [], {}
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for x, y in dl["test"]:
            logits = model(x.to(dev))
            pred = logits.argmax(1).cpu()
            for i in range(len(y)):
                c = int(y[i])
                if per_class.get(c, 0) < 1 and int(pred[i]) == c:
                    picked.append((x[i].clone(), c))
                    per_class[c] = 1
            if len(picked) >= a.n:
                break
    picked = picked[: a.n]

    cam = GradCAM(model=model, target_layers=[target_layer(model, a.arch)])
    fig, axes = plt.subplots(2, (len(picked) + 1) // 2, figsize=(4 * ((len(picked) + 1) // 2), 8.4), dpi=140)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, (t, c) in enumerate(picked):
        img = denormalize(t.unsqueeze(0), "imagenet")[0]
        rgb = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gray = torch.tensor(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)).float().unsqueeze(0).repeat(3, 1, 1) / 255.0
        heat = cam(input_tensor=gray.unsqueeze(0).to(dev), targets=[ClassifierOutputTarget(c)])[0]
        ax = axes[k]
        ax.imshow(rgb)
        ax.imshow(heat, cmap="jet", alpha=0.45)
        ax.set_title(f"{classes[c]}（真值/预测一致）", fontsize=10)
        ax.axis("off")
    cam.__del__() if hasattr(cam, "__del__") else None
    fig.suptitle(f"{a.exp} · Grad-CAM（红=模型关注区）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "outputs" / "best_gradcam.png"
    fig.savefig(out, dpi=140, facecolor="white")
    plt.close(fig)
    print(f"saved {out}  ({len(picked)} 张)")


if __name__ == "__main__":
    main()

