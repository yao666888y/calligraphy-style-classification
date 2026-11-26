# -*- coding: utf-8 -*-
"""合成「模拟手机照片」评测集：官方测试集 → 带纸色/米字格/台灯阴影/模糊的手机照片。

只生成一次，存到 data/phone_sim/raw/*.png + labels.csv；
评测时各模型必须再走自己的拍照清洗流水线。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "phone_sim"
N_PER_CLASS = 100
SEED = 2024


def simulate(src: Path, rng: np.random.RandomState) -> Image.Image:
    """先在纸上画格、再让字压在格上（真实书写顺序），最后加阴影/模糊/JPEG。"""
    with Image.open(src) as im:
        glyph = np.asarray(im.convert("L").resize((640, 640), Image.BICUBIC), dtype=np.float32)
    h, w = glyph.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # 1) 纸色 + 台灯阴影
    base = np.stack([
        252 - rng.uniform(0, 14) - 12 * (yy / h),
        247 - rng.uniform(0, 14) - 14 * (yy / h),
        228 - rng.uniform(0, 14) - 16 * (yy / h),
    ], -1)
    paper = np.broadcast_to(base, (h, w, 3)).copy()
    cx, cy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.15, 0.7) * h
    shadow = 1.0 - 0.35 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (0.45 * w) ** 2)))
    paper *= shadow[..., None]

    # 2) 先画页框 + 米字格（在字的下层）
    dark = rng.uniform(35, 90)
    paper[:3, :] = dark
    paper[-3:, :] = dark
    paper[:, :3] = dark
    paper[:, -3:] = dark
    paper[:, w // 2:w // 2 + 2] = dark + 40
    paper[h // 2:h // 2 + 2, :] = dark + 40
    if rng.random() < 0.7:
        step = int(rng.choice([8, 12, 16]))
        for k in range(0, w, step * 2):
            paper[k:k + step, k:k + 2] = dark + 30
            paper[k:k + 2, k:k + step] = dark + 30
            paper[k:k + step, w - k - 2:w - k] = dark + 30
            paper[h - k - 2:h - k, k:k + step] = dark + 30

    # 3) 字压在格子上（alpha = 墨迹浓度）
    alpha = np.clip((255.0 - glyph) / 255.0, 0, 1)[..., None]
    ink_rgb = np.array([rng.uniform(15, 40), rng.uniform(15, 40), rng.uniform(15, 40)], dtype=np.float32)
    a = paper * (1 - alpha) + ink_rgb * alpha

    # 4) 模糊 + 噪声 + JPEG 痕迹
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), rng.uniform(0.6, 1.8))
    a += rng.normal(0, rng.uniform(2, 6), a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.randint(45, 85)))
    buf.seek(0)
    with Image.open(buf) as im2:
        return im2.convert("RGB")


def main() -> None:
    rng = np.random.RandomState(SEED)
    test_root = ROOT / "data" / "test_datasets" / "test"
    classes = sorted(d for d in test_root.iterdir() if d.is_dir())
    (OUT / "raw").mkdir(parents=True, exist_ok=True)

    rows = []
    idx = 0
    for label, cdir in enumerate(classes):
        files = sorted(cdir.glob("*.jpg"))
        pick = rng.choice(len(files), min(N_PER_CLASS, len(files)), replace=False)
        for k in pick:
            src = files[int(k)]
            out = simulate(src, rng)
            name = f"{idx:05d}.png"
            out.save(OUT / "raw" / name)
            rows.append((name, label, classes[label if False else label].name, str(src.relative_to(ROOT))))
            idx += 1
        print(f"  {cdir.name}: {len(pick)} 张 (累计 {idx})", flush=True)

    with (OUT / "labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "class_dir", "source"])
        w.writerows(rows)
    print(f"完成: {len(rows)} 张 → {OUT}")


if __name__ == "__main__":
    main()
