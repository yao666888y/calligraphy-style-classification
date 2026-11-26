# -*- coding: utf-8 -*-
"""模型病理体检：全白 / 全黑 / 随机噪点 / 纯黑方块，看模型会不会「瞎猜」。

    python scripts/check_model_pathology.py --exp exp1_resnet18_base --arch resnet18 \
        --fill 0.68 --binarize 0 --out outputs/pathology_exp1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import build_eval_transforms  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp1_resnet18_base")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fill", type=float, default=0.68)
    ap.add_argument("--binarize", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ckpt = Path(a.ckpt) if a.ckpt else ROOT / "checkpoints" / a.exp / "best.pth"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.arch, num_classes=len(classes), pretrained=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    tf = build_eval_transforms(a.img_size, "imagenet")

    def probe(img: Image.Image, tag: str) -> dict:
        try:
            out = normalize_calligraphy_image(img, size=a.img_size, fill=a.fill,
                                              binarize=bool(a.binarize), mask_background="auto")
        except ValueError:
            # 全白这类图没有墨迹，清洗会拒绝；体检时退回「直接缩放」看模型反应
            out = img.convert("RGB").resize((a.img_size, a.img_size), Image.LANCZOS)
        t = tf(out).unsqueeze(0).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            prob = torch.softmax(model(t).float(), 1)[0].cpu()
        v, i = prob.topk(3)
        rec = {"input": tag, "top1": classes[int(i[0])], "prob": round(float(v[0]), 4),
               "top3": [{"name": classes[int(j)], "prob": round(float(s), 4)} for s, j in zip(v, i)]}
        print(f"  {tag:<6s} → {rec['top1']} {rec['prob']:.3f}", flush=True)
        return rec

    rng = np.random.RandomState(0)
    square = np.full((128, 128, 3), 255, np.uint8)
    square[40:90, 40:90] = 0
    probes = [
        probe(Image.new("RGB", (128, 128), (255, 255, 255)), "全白"),
        probe(Image.new("RGB", (128, 128), (0, 0, 0)), "全黑"),
        probe(Image.fromarray(rng.randint(0, 255, (128, 128, 3), dtype=np.uint8)), "噪点"),
        probe(Image.fromarray(square), "黑方块"),
    ]
    out = Path(a.out) if a.out else ROOT / "outputs" / f"pathology_{a.exp}.json"
    out.write_text(json.dumps({"exp": a.exp, "probes": probes}, ensure_ascii=False, indent=2),
                   encoding="utf-8-sig")
    print(f"  [saved] {out}")


if __name__ == "__main__":
    main()
