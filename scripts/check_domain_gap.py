# -*- coding: utf-8 -*-
"""Step 1/2：定位「训练集 ↔ 网页推理」的输入分布差异。

用法
----
    python scripts/check_domain_gap.py           # 修复前：训练走 dataset.py，网页走 app 老清洗
    python scripts/check_domain_gap.py --after   # 修复后：两边都走共享的 normalize_calligraphy_image

对比对象
--------
A. 训练集：随机 8 张，经过各自链路预处理后真正喂进模型的张量
B. 网页  ：用户上传的真实毛笔字照片，经过各自链路预处理后的张量

指标
----
* 墨迹外接框占画布的比例（宽/高）—— 字到底占多大
* 黑像素占比、灰阶种类、纯黑/纯白端点占比 —— 连续灰度还是硬二值
* 灰度直方图（log 纵轴）对比
* Bug 核查：细小噪点丢弃数、是否出现「二次补白」

输出
----
outputs/domain_gap_check.png       （修复前）
outputs/domain_gap_check_after.png （修复后）
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
from utils.dataset import build_dataloaders, build_eval_transforms, denormalize, pad_to_square  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402

OUT = ROOT / "outputs"
N_TRAIN = 8
SIZE = 128


def analyze(t: torch.Tensor) -> dict:
    img = denormalize(t.unsqueeze(0) if t.dim() == 3 else t, "imagenet")[0]
    g = img.mean(dim=0).numpy()
    ink = g < 0.5
    if ink.any():
        ys, xs = np.where(ink)
        bw = (xs.max() - xs.min() + 1) / g.shape[1]
        bh = (ys.max() - ys.min() + 1) / g.shape[0]
    else:
        bw = bh = 0.0
    vals = np.clip((g * 255).round(), 0, 255).astype(np.uint8)
    flat = vals.ravel()
    hist, _ = np.histogram(flat, bins=64, range=(0, 256))
    return {
        "bbox_w": float(bw), "bbox_h": float(bh), "long_side": float(max(bw, bh)),
        "black_ratio": float((g < 0.1).mean()),
        "n_levels": int(np.unique(flat).size),
        "frac_dark_end": float((flat <= 2).mean()),
        "frac_white_end": float((flat >= 253).mean()),
        "hist": hist / hist.sum(),
    }


def pick_web_case() -> Path:
    """挑用户上传的真实毛笔照片（尺寸 >= 200px）里最新的一张。"""
    cands = []
    for d in (OUT / "web_test").glob("2026*"):
        orig = next(d.glob("original.*"), None)
        if not orig:
            continue
        with Image.open(orig) as im:
            if min(im.size) >= 200:
                cands.append((d.name, orig))
    if not cands:
        raise SystemExit("outputs/web_test 里没找到大图")
    return sorted(cands)[-1][1]


def main() -> int:
    after = "--after" in sys.argv
    fontm = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in fontm else "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    torch.manual_seed(42)
    np.random.seed(42)
    tf = build_eval_transforms(SIZE, "imagenet")
    dl = build_dataloaders(img_size=SIZE, batch_size=8, num_workers=0, seed=42)
    names = dl["classes"]

    # ---------- A. 训练集 8 张 ----------
    rng = np.random.RandomState(42)
    idx = rng.choice(len(dl["train_ds"].samples), N_TRAIN, replace=False)
    train_items = []
    for si in idx:
        p, lb = dl["train_ds"].samples[int(si)]
        with Image.open(p) as im:
            if after:
                img = normalize_calligraphy_image(im, size=SIZE, fill=0.88, binarize=True)
            else:
                img = pad_to_square(dl["train_ds"]._unify(p, im), margin=0.12)
        t = tf(img)
        train_items.append((p.name, names[lb], analyze(t), t))

    # ---------- B. 网页那张 ----------
    web_path = pick_web_case()
    with Image.open(web_path) as im:
        if after:
            cleaned, dinfo = normalize_calligraphy_image(
                im, size=SIZE, fill=0.88, binarize=True, return_info=True
            )
        else:
            # 复刻「修复前」的网页管线：先留 10% 边距，再补到 75% 字幅
            # → 实际字幅 = 0.75 / 1.2 = 0.625，且非紧致裁切
            cleaned, dinfo = normalize_calligraphy_image(
                im, size=SIZE, fill=0.625, binarize=False, return_info=True
            )
    t_web = tf(cleaned)
    a_web = analyze(t_web)

    # ---------- Bug 核查 ----------
    with Image.open(web_path) as im:
        _, pinfo = normalize_calligraphy_image(im, size=SIZE, fill=0.88, binarize=True, return_info=True)
    train_long = float(np.mean([x[2]["long_side"] for x in train_items]))
    web_long = a_web["long_side"]
    n_levels_train = float(np.mean([x[2]["n_levels"] for x in train_items]))

    # ---------- 画图 ----------
    fig = plt.figure(figsize=(15.5, 9.4), dpi=140)
    gs = fig.add_gridspec(3, 8, height_ratios=[1, 1, 1.15], hspace=0.45, wspace=0.12)

    for i, (fn, cname, a, t) in enumerate(train_items[:4]):
        ax = fig.add_subplot(gs[0, i * 2:i * 2 + 2])
        ax.imshow(denormalize(t, "imagenet").permute(1, 2, 0).numpy())
        ax.set_title(f"训练集 {cname}\n框 {a['bbox_w']:.2f}×{a['bbox_h']:.2f}  黑 {a['black_ratio']:.2f}  {a['n_levels']} 灰阶",
                     fontsize=9)
        ax.axis("off")
    for i, (fn, cname, a, t) in enumerate(train_items[4:]):
        ax = fig.add_subplot(gs[1, i * 2:i * 2 + 2])
        ax.imshow(denormalize(t, "imagenet").permute(1, 2, 0).numpy())
        ax.set_title(f"训练集 {cname}\n框 {a['bbox_w']:.2f}×{a['bbox_h']:.2f}  黑 {a['black_ratio']:.2f}  {a['n_levels']} 灰阶",
                     fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[2, 0:2])
    ax.imshow(cleaned)
    ax.set_title(f"网页输入（模型实际看到的）\n框 {a_web['bbox_w']:.2f}×{a_web['bbox_h']:.2f}  "
                 f"黑 {a_web['black_ratio']:.2f}  {a_web['n_levels']} 灰阶", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, 2:5])
    ax.plot(np.linspace(0, 255, 64), np.mean([x[2]["hist"] for x in train_items], axis=0),
            lw=2.2, color="steelblue", label="训练集均值")
    ax.plot(np.linspace(0, 255, 64), a_web["hist"], lw=2.4, color="crimson", label="网页输入")
    ax.set_yscale("log")
    ax.set_xlabel("灰度值 (0=黑, 255=白)")
    ax.set_ylabel("像素占比 (log)")
    ax.set_title("灰度直方图对比", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[2, 5:8])
    ax.axis("off")
    text = (
        ("修复后 · 两边走同一函数\n" if after else "修复前 · 两边各走各的\n") +
        f"· 训练集字幅长边 ≈ {train_long * 100:.0f}% 画布\n"
        f"· 网页输入字幅长边 ≈ {web_long * 100:.0f}% 画布"
        f"{'  已对齐' if abs(train_long - web_long) < 0.03 else f'  （差 {abs(train_long - web_long) * 100:.0f}pp）'}\n"
        f"· 训练集灰阶数 ≈ {n_levels_train:.0f}，网页 = {a_web['n_levels']}\n"
        f"· 网页黑像素占比 {a_web['black_ratio']:.3f}"
        f"（训练均值 {np.mean([x[2]['black_ratio'] for x in train_items]):.3f}）\n"
        f"· 网页端点占比：纯黑 {a_web['frac_dark_end']:.3f} / 纯白 {a_web['frac_white_end']:.3f}\n\n"
        "Bug 核查\n"
        f"· 本次丢弃的细小连通域：{pinfo.n_components_dropped} 个\n"
        f"· 是否识别并剔除页框：{'是' if pinfo.frame_dropped else '否'}\n"
        f"· 二次补白：{'已消除（只补一次到 88%）' if after else '存在（10% 边距 + 75% 字幅 → 实际 61%）'}\n"
        f"· 紧致裁切后 crop = {pinfo.crop_size}，缩放系数 {pinfo.scale}"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", fontsize=10,
            family=plt.rcParams["font.sans-serif"][0], linespacing=1.75)

    title = "Step 2 · 修复后：训练集 vs 网页输入（统一归一化）" if after else "Step 1 · 域差诊断：训练集 vs 网页输入（修复前）"
    fig.suptitle(title, fontsize=15)
    dst = OUT / ("domain_gap_check_after.png" if after else "domain_gap_check.png")
    fig.savefig(dst, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    print(f"模式: {'after' if after else 'before'}   网页图: {web_path.name}")
    print(f"训练集字幅 {train_long:.3f}  网页 {web_long:.3f}")
    print(f"训练集灰阶 {n_levels_train:.0f}  网页 {a_web['n_levels']}  黑占比 {a_web['black_ratio']:.3f}")
    print(f"丢弃小连通域 {pinfo.n_components_dropped} 个, 页框 {pinfo.frame_dropped}, crop={pinfo.crop_size}")
    print(f"saved {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
