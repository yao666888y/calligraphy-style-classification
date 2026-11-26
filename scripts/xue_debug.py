# -*- coding: utf-8 -*-
"""「雪」字定点对照：4 种预处理组合下 exp1 / exp6 的 Top-5 与颜真卿排名。

① exp1 原样（照片入口 web_v1：0.88 字幅 + 非二值 + 掩膜刷白）
② exp6 直接（0.88 字幅 + 二值化，224）
③ exp6 + 先降到 64×64 再拉回 224
④ ③ 之前先填掉面积 < 0.15% 的孤立白色孔洞（石花/飞白），保留正常封闭留白

输出：outputs/xue_debug.png、outputs/xue_debug.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image
from torchvision import transforms

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import NORM_STATS, build_eval_transforms  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402

XUE = ROOT / "outputs" / "web_test" / "20261125_185703_012" / "original.png"
SIZE = 224
_mean, _std = NORM_STATS["imagenet"]
TF224 = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std)])


def load(path: Path, arch: str, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(arch, num_classes=len(ck["classes"]), pretrained=False).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck["classes"]


def topk(model, tensor, names, dev, k=5):
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
        p = torch.softmax(model(tensor.unsqueeze(0).to(dev)).float(), 1)[0].cpu()
    v, i = p.topk(min(k, len(names)))
    rank = int((p.argsort(descending=True) == names.index("颜真卿")).nonzero()) + 1
    return [(names[int(j)], round(float(s), 4)) for s, j in zip(v, i)], rank, round(float(p[names.index("颜真卿")]), 4)


def fill_small_holes(img: Image.Image, ratio: float = 0.0015) -> tuple[Image.Image, int]:
    """填充「被墨包围、且面积 < ratio×整图」的白色孔洞（石花/飞白白点）。"""
    a = np.asarray(img.convert("L"))
    ink = a < 128
    inv = (~ink).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = a.shape
    out = ink.copy()
    filled = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area >= ratio * h * w:
            continue
        if x <= 0 or y <= 0 or x + ww >= w or y + hh >= h:
            continue                      # 触边的是背景，不动
        out[labels == i] = True
        filled += 1
    return Image.fromarray(np.where(out, 0, 255).astype(np.uint8)).convert("RGB"), filled


def main() -> None:
    fm = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in fm else "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m1, n1 = load(ROOT / "checkpoints" / "exp1_resnet18_base" / "best.pth", "resnet18", dev)
    m6, n6 = load(ROOT / "checkpoints" / "exp6_convnext_224" / "best.pth", "convnext_tiny", dev)
    tf1 = build_eval_transforms(128, "imagenet")

    raw = Image.open(XUE).convert("RGB")

    # ① exp1：网页照片入口那条（0.88 字幅、非二值、掩膜 auto）
    img1 = normalize_calligraphy_image(raw, size=128, fill=0.88, binarize=False, mask_background="auto")
    t1 = tf1(img1)
    # ② exp6 直接
    img2 = normalize_calligraphy_image(raw, size=SIZE, fill=0.88, binarize=True)
    t2 = TF224(img2)
    # ③ 64 往返
    img3 = img2.resize((64, 64), Image.LANCZOS).resize((SIZE, SIZE), Image.LANCZOS)
    t3 = TF224(img3)
    # ④ 先填小孔，再 64 往返
    img4_filled, n_fill = fill_small_holes(img2)
    img4 = img4_filled.resize((64, 64), Image.LANCZOS).resize((SIZE, SIZE), Image.LANCZOS)
    t4 = TF224(img4)

    print(f"图片: {XUE.relative_to(ROOT)}  填充孔洞 {n_fill} 个")
    res = {}
    for tag, model, names, t in (
        ("① exp1 原样", m1, n1, t1),
        ("② exp6 直接", m6, n6, t2),
        ("③ exp6 + 64往返", m6, n6, t3),
        ("④ ③+填孔", m6, n6, t4),
    ):
        top, rank, prob = topk(model, t, names, dev, 5)
        res[tag] = {"top5": top, "yzq_rank": rank, "yzq_prob": prob}
        print(f"{tag:<18s} 颜真卿 第{rank}名 {prob:.4f}   Top5: "
              + ", ".join(f"{n}:{p:.3f}" for n, p in top), flush=True)

    # ---------- 拼图 ----------
    yzq_dir = ROOT / "data" / "train_datasets" / "train" / "yzq"
    smh_dir = ROOT / "data" / "train_datasets" / "train" / "smh"
    rng = np.random.RandomState(0)
    yzq = [sorted(yzq_dir.glob("*.jpg"))[i] for i in rng.choice(len(list(yzq_dir.glob("*.jpg"))), 4, replace=False)]
    smh_all = sorted(smh_dir.glob("*.jpg"))
    smh = [smh_all[i] for i in rng.choice(len(smh_all), 4, replace=False)]

    fig, axes = plt.subplots(3, 4, figsize=(12.4, 10.2), dpi=130)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(raw.resize((300, 300), Image.LANCZOS))
    axes[0, 0].set_title("雪 · 上传原图", fontsize=10)
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title("② 清洗（exp6 直接输入）", fontsize=10)
    axes[0, 2].imshow(img4_filled)
    axes[0, 2].set_title(f"④ 填孔后（{n_fill} 个小孔）", fontsize=10)
    axes[0, 3].axis("off")
    axes[0, 3].text(0.0, 0.95,
                    "颜真卿排名/概率\n" + "\n".join(
                        f"{k.split(' ')[0]} 第{v['yzq_rank']}名 {v['yzq_prob']:.3f}" for k, v in res.items()),
                    fontsize=9, va="top")

    for i, p in enumerate(yzq):
        with Image.open(p) as im:
            axes[1, i].imshow(im.convert("RGB").resize((224, 224), Image.LANCZOS))
        axes[1, i].set_title("颜真卿 原图" if i == 0 else "", fontsize=9)
    for i, p in enumerate(smh):
        with Image.open(p) as im:
            axes[2, i].imshow(im.convert("RGB").resize((224, 224), Image.LANCZOS))
        axes[2, i].set_title("沙孟海 原图" if i == 0 else "", fontsize=9)

    fig.suptitle("「雪」字对照：清洗前后 & 颜真卿 / 沙孟海 训练样本", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "outputs" / "xue_debug.png"
    fig.savefig(out, dpi=130, facecolor="white")
    plt.close(fig)
    (ROOT / "outputs" / "xue_debug.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()

