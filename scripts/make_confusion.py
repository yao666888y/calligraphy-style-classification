# -*- coding: utf-8 -*-
"""生成中文混淆矩阵（官方对齐测试集）。

    python scripts/make_confusion.py --exp exp2_domain_aligned --arch resnet18 --img_size 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import build_dataloaders  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

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
    preds, trues = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for x, y in dl["test"]:
            preds.append(model(x.to(dev)).argmax(1).cpu())
            trues.append(y)
    p, t = torch.cat(preds).numpy(), torch.cat(trues).numpy()

    cm = confusion_matrix(t, p, labels=list(range(len(classes))))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(13, 11), dpi=150)
    im = ax.imshow(cmn, cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("预测书家", fontsize=12)
    ax.set_ylabel("真实书家", fontsize=12)
    ax.set_title(f"{a.exp} · 官方测试集混淆矩阵（行归一化，acc={np.trace(cmn) / len(classes):.4f}）", fontsize=14)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v * 100:.0f}" if v < 0.995 else "100",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "#3a2f22")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="比例")
    fig.tight_layout()
    out = Path(a.out) if a.out else ROOT / "outputs" / "best_confusion_matrix.png"
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"saved {out}  acc={np.trace(cmn) / len(classes):.4f}")


if __name__ == "__main__":
    main()

