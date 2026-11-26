# -*- coding: utf-8 -*-
"""用 6 张带标签的真实照片重测「照片入口」清洗 + exp6 预测。

输出：
  outputs/labeled_recheck.png   每张的清洗图（带清洗前后墨迹占比与 Top-3）
  outputs/labeled_recheck.json  明细
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
from utils.dataset import NORM_STATS  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402

CKPT = ROOT / "checkpoints" / "exp6_convnext_224" / "best.pth"
SIZE = 224
LABELS = ROOT / "outputs" / "web_test" / "labels.csv"
_mean, _std = NORM_STATS["imagenet"]
TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std)])


def main() -> None:
    m = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in m else "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    names = ck["classes"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("convnext_tiny", num_classes=len(names), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    rows = list(csv.DictReader((LABELS.open(encoding="utf-8-sig"))))
    picked = [r for r in rows if r["真实书家"].strip()]

    fig, axes = plt.subplots(len(picked), 2, figsize=(9.2, len(picked) * 4.0), dpi=120)
    axes = np.atleast_2d(axes)
    out_rows = []
    for k, r in enumerate(picked):
        p = ROOT / "outputs" / "web_test" / r["文件名"]
        with Image.open(p) as im:
            raw = im.convert("RGB")
            cleaned, info = normalize_calligraphy_image(
                raw, size=SIZE, fill=0.88, binarize=True, return_info=True
            )
        t = TF(cleaned).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            prob = torch.softmax(model(t).float(), 1)[0].cpu()
        v, i = prob.topk(3)
        delta = info.ink_mask_ratio - info.ink_src_ratio
        warn = abs(delta) > 0.03

        small = raw.copy()
        small.thumbnail((520, 520))
        axes[k, 0].imshow(small)
        axes[k, 0].set_title(f"{r['文件名'].split('/')[0].split('_')[-1]} 原图（{r['来源类型']}）", fontsize=9)
        axes[k, 1].imshow(cleaned)
        tag = "  ⚠偏差>3pp" if warn else ""
        axes[k, 1].set_title(
            f"清洗后 墨迹 {info.ink_src_ratio * 100:.1f}%→{info.ink_mask_ratio * 100:.1f}%"
            f"（Δ{delta * 100:+.1f}pp）{tag}\n字幅 {info.glyph_fill_actual * 100:.0f}%  "
            f"dark_bg={info.dark_bg}", fontsize=8.5)
        for ax in (axes[k, 0], axes[k, 1]):
            ax.axis("off")

        rec = {
            "file": r["文件名"],
            "truth": r["真实书家"],
            "source": r["来源类型"],
            "dark_bg": info.dark_bg,
            "ink_src_ratio": info.ink_src_ratio,
            "ink_mask_ratio": info.ink_mask_ratio,
            "ink_final_ratio": info.ink_final_ratio,
            "delta_pp": round(delta * 100, 2),
            "warn": warn,
            "glyph_fill_actual": info.glyph_fill_actual,
            "stroke_px": info.stroke_px,
            "top3": [{"name": names[int(j)], "prob": round(float(s), 4)} for s, j in zip(v, i)],
        }
        out_rows.append(rec)
        print(f"{r['文件名']:<40s} 真值={r['真实书家']:<4s} {r['来源类型']:<5s} "
              f"dark={info.dark_bg!s:<5s} 墨 {info.ink_src_ratio * 100:5.1f}%→{info.ink_mask_ratio * 100:5.1f}% "
              f"Δ{delta * 100:+5.1f}pp 字幅 {info.glyph_fill_actual * 100:3.0f}%  "
              f"-> " + ", ".join(f"{x['name']}:{x['prob']:.2f}" for x in rec["top3"])
              + ("  [RED]" if warn else ""), flush=True)

    fig.suptitle("带标签真实照片：清洗后输入（exp6 / 224）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    png = ROOT / "outputs" / "labeled_recheck.png"
    fig.savefig(png, dpi=120, facecolor="white")
    plt.close(fig)
    (ROOT / "outputs" / "labeled_recheck.json").write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"\n[saved] {png} / labeled_recheck.json")


if __name__ == "__main__":
    main()
