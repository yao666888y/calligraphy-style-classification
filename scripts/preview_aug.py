# -*- coding: utf-8 -*-
"""Step 2 校验脚本：训练集 DataLoader 取 1 个 batch(16 张)，反归一化后拼图。

输出:
  outputs/aug_preview.png   4×4 网格（torchvision.utils.make_grid 拼接，保证不变形）
  outputs/aug_check_log.txt 运行日志

同时打印: 张量形状/数值范围/标签索引、笔画越界检查、增强组成。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.dataset import (  # noqa: E402
    build_dataloaders,
    build_eval_transforms,
    build_train_transforms,
    CalligraphyDataset,
    denormalize,
)

OUT = ROOT / "outputs"
BATCH, IMG_SIZE, NROW, PAD, NORM, SEED = 16, 128, 4, 30, "imagenet", 42


def setup_font() -> str:
    cands = ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((c for c in cands if c in have), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [pick] + cands
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return pick


def ink_margins(t: torch.Tensor, thr: float = 0.90) -> dict:
    """t:[C,H,W] in [0,1] → 墨迹到四边的距离（px）。"""
    g = t.mean(dim=0).numpy()
    mask = g < thr
    if not mask.any():
        return {"left": 999, "right": 999, "top": 999, "bottom": 999}
    ys, xs = np.where(mask)
    h, w = g.shape
    return {
        "left": int(xs.min()),
        "right": int(w - 1 - xs.max()),
        "top": int(ys.min()),
        "bottom": int(h - 1 - ys.max()),
    }


def main() -> int:
    font = setup_font()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("=" * 78)
    print("Step 2 数据管道校验（修正版：增强不裁笔画 + make_grid 拼图）")
    print("=" * 78)
    print(f"中文字体: {font}")

    dl = build_dataloaders(img_size=IMG_SIZE, batch_size=BATCH, num_workers=2, seed=SEED, norm=NORM)
    s, names = dl["stats"], dl["classes"]

    print("\n[1] 划分")
    print(f"    train / val / test = {s['n_train']} / {s['n_val']} / {s['n_test']}   (val={s['val_ratio']:.0%}, seed={SEED})")
    print(f"    白边 margin = {s['margin']:.0%}，缩放∈{s['scale_range'] if 'scale_range' in s else '(0.95,1.05)'}，平移 4%")

    # 取 batch，并带上路径便于逐张对照
    dl["train_ds"].return_path = True
    batch = next(iter(dl["train"]))
    images, labels, paths = batch
    dl["train_ds"].return_path = False

    print("\n[2] 训练集 batch")
    print(f"    张量形状    : {tuple(images.shape)}   dtype={images.dtype}")
    print(f"    归一化范围  : [{images.min():.4f}, {images.max():.4f}]  mean={images.mean():.4f} std={images.std():.4f}")
    img = denormalize(images, NORM)
    print(f"    反归一化范围: [{img.min():.4f}, {img.max():.4f}]")
    print(f"    标签索引    : {labels.tolist()}")
    print(f"    对应类别    : {[names[i] for i in labels.tolist()]}")

    print("\n[3] 笔画越界检查（关键：确认增强没把字切掉）")
    ev = CalligraphyDataset(
        list(zip(paths, labels.tolist())),
        build_eval_transforms(IMG_SIZE, NORM),
        bg_mode="auto",
        img_size=IMG_SIZE,
        margin=s["margin"],
        cache_bg=False,
    )
    print(f"    {'#':>2} {'类别':<6s} {'增强后 左/右/上/下':>20s} {'原图 左/右/上/下':>20s}  判定")
    bad = []
    for i in range(BATCH):
        m_aug = ink_margins(img[i])
        te, _ = ev[i]
        m_raw = ink_margins(denormalize(te, NORM))
        clipped = min(m_aug.values()) <= 0
        if clipped:
            bad.append(i)
        flag = "被裁!" if clipped else ("原图即贴边" if min(m_raw.values()) <= 0 else "OK")
        print(f"    {i:>2} {names[labels[i].item()]:<6s} "
              f"{m_aug['left']:>4},{m_aug['right']:>3},{m_aug['top']:>3},{m_aug['bottom']:>3}       "
              f"{m_raw['left']:>4},{m_raw['right']:>3},{m_raw['top']:>3},{m_raw['bottom']:>3}      {flag}")
    print(f"\n    增强后仍被裁的图: {bad if bad else '无 —— 全部笔画完整'}")

    print("\n[4] 增强组成")
    for i, t in enumerate(build_train_transforms(allow_hflip=False).transforms, 1):
        print(f"    {i:>2}. {t}")

    # ---------- [5] make_grid 拼图 ----------
    # PAD 加大到 30：灰间隔里要放得下每格的类名标签
    grid = make_grid(img, nrow=NROW, padding=PAD, pad_value=0.85)  # [3,H,W]
    grid_np = grid.permute(1, 2, 0).numpy()
    gh, gw = grid.shape[1], grid.shape[2]

    # 图像 axes 物理尺寸做成正方形 → 每格严格 128×128，不可能被拉伸
    fig_w = 10.9
    img_in = fig_w * 0.96
    top_h, bot_h = 0.55, 0.30
    fig_h = img_in + top_h + bot_h
    img_frac_h = img_in / fig_h
    img_bottom = bot_h / fig_h
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = fig.add_axes([0.02, bot_h / fig_h, 0.96, img_in / fig_h])
    ax.imshow(grid_np, interpolation="lanczos", aspect="equal")
    ax.set_axis_off()

    # 标签画在每行上方的灰色间隔里（imshow 的 origin=upper，y 向下增大）
    for i in range(BATCH):
        r, c = divmod(i, NROW)
        cell_left = PAD + c * (IMG_SIZE + PAD)
        cell_top = PAD + r * (IMG_SIZE + PAD)
        ax.text(
            cell_left + IMG_SIZE / 2,
            cell_top - 9,
            f"{i}: {names[labels[i].item()]}",
            ha="center", va="bottom", fontsize=11.5, color="#222222",
        )

    fig.suptitle(
        "Step 2 · 训练集单 Batch（16 张，增强后 128×128，白底黑字，无翻转）",
        fontsize=19, y=0.985,
    )
    fig.text(
        0.5, 0.008,
        "make_grid 拼接（浅灰为图间距，每格严格 128×128 正方形）｜±8° 旋转(expand) + 缩放0.95~1.05 + 平移4% + 轻量擦除",
        ha="center", fontsize=9, color="#555555",
    )
    path = OUT / "aug_preview.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)

    from PIL import Image

    with Image.open(path) as im:
        print(f"\n[out] {path}  ({im.size[0]}x{im.size[1]})")
    print(f"[out] grid 张量 {tuple(grid.shape)}，理论边长 {gw}px（=4·128+5·{PAD}）")

    # ---------- [6] 自检 ----------
    # 6.1 几何：make_grid 的每一格必须正好 128×128，且落在预期偏移上
    cell0 = grid[:, PAD:PAD + IMG_SIZE, PAD:PAD + IMG_SIZE]
    geom_ok = torch.allclose(cell0, img[0], atol=1e-6)
    assert gw == NROW * IMG_SIZE + (NROW + 1) * PAD, "grid 尺寸与 4·128+5·30 不符"
    print(f"\n[6] 自检")
    print(f"    网格张量 {gw}×{gh}px = 4×128 + 5×{PAD}；第 0 格与 img[0] 像素一致: {geom_ok}")
    assert geom_ok

    # 6.2 渲染：整块网格在 PNG 里必须是正方形（若被拉伸，灰度块的 bbox 就不是正方形）
    with Image.open(path) as im:
        W, H = im.size
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.int16)
    y0 = int(round((1 - (img_bottom + img_frac_h)) * H))
    y1 = int(round((1 - img_bottom) * H))
    x0, x1 = int(round(0.02 * W)), int(round(0.98 * W))
    sub = arr[y0:y1, x0:x1]
    ys, xs = np.where(sub < 250)          # 灰间隔+墨迹都会 <250，白底为 255
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    ratio = bh / bw
    print(f"    网格在 PNG 中的实际尺寸 {bw}×{bh}px，高宽比 {ratio:.4f} "
          f"→ {'正方形 OK，无拉伸' if abs(ratio - 1) <= 0.01 else '仍有变形!'}")
    print(f"    推算每格渲染尺寸 ≈ {bw / (NROW * IMG_SIZE + (NROW + 1) * PAD) * IMG_SIZE:.1f} × "
          f"{bh / (NROW * IMG_SIZE + (NROW + 1) * PAD) * IMG_SIZE:.1f} px")

    # 6.3 按你的要求单独导出 3 / 12 / 15 号单图（128 原尺寸 + 512 放大便于看细节）
    print("\n[7] 单图导出（反归一化后）")
    for idx in (3, 12, 15):
        arr = (img[idx].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        p128 = OUT / f"single_{idx:02d}.png"
        p512 = OUT / f"single_{idx:02d}_x4.png"
        Image.fromarray(arr).save(p128)
        Image.fromarray(arr).resize((512, 512), Image.NEAREST).save(p512)
        print(f"    #{idx:>2} {names[labels[idx].item()]:<4s} → {p128.name} ({tuple(arr.shape)}) / {p512.name}")

    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
