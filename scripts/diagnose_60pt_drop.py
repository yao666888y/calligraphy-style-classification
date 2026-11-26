# -*- coding: utf-8 -*-
"""诊断：exp6 官方 98.59% → phone_sim 38.70% 的 60 个点丢在哪。

四件事
------
1. 同字清洗前后并排对比（10 张 clean88 误判字 vs 其 aligned_cache 原字）
2. 三组图像的 4 个分布指标：黑像素占比、平均笔画宽、Canny 边缘比、字幅占比
3. 零成本盲测 A/B/C：A=224 直入；B=先降到 64 再拉回 224；C=形态学纠到训练集平均墨量
4. 10 张真实手写的完整排名诊断

输出：outputs/cache_vs_clean88.png、outputs/diagnose_60pt_drop.json
"""
from __future__ import annotations

import base64  # noqa: F401
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# Windows 上 torch 与 matplotlib 各带一份 OpenMP 运行库，会互相冲突
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
from utils.dataset import NORM_STATS  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402

CKPT = ROOT / "checkpoints" / "exp6_convnext_224" / "best.pth"
SIZE = 224
N_STAT = 500
SEED = 2024

# 已知真值的真实手写样本（用例目录序号 → 书家）
KNOWN_TRUTH = {"003": "欧阳询", "004": "颜真卿"}

_mean, _std = NORM_STATS["imagenet"]
TF = transforms.Compose([transforms.ToTensor(), transforms.Normalize(_mean, _std)])


def setup_font():
    m = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei" if "Microsoft YaHei" in m else "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_model():
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    names = ck["classes"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model("convnext_tiny", num_classes=len(names), pretrained=False).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, names, dev


@torch.no_grad()
def predict_probs(model, imgs: list[Image.Image], dev) -> np.ndarray:
    out = []
    for i in range(0, len(imgs), 64):
        chunk = torch.stack([TF(im.convert("RGB")) for im in imgs[i:i + 64]]).to(dev)
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            logits = model(chunk)
        out.append(torch.softmax(logits.float(), 1).cpu())
    return torch.cat(out).numpy()


def metrics(img: Image.Image) -> dict:
    """在 128 画布上算 4 个指标。"""
    g = np.asarray(img.convert("L").resize((128, 128), Image.LANCZOS), dtype=np.uint8)
    ink = g < 128
    if not ink.any():
        return {"black": 0.0, "stroke": 0.0, "edge": 0.0, "bbox": 0.0}
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    stroke = float(2.0 * dist[ink].mean())
    edges = cv2.Canny(g, 50, 150)
    ys, xs = np.where(ink)
    bbox = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) / 128.0
    return {
        "black": float(ink.mean()),
        "stroke": stroke,
        "edge": float((edges > 0).mean()),
        "bbox": float(bbox),
    }


def mask_of(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((128, 128), Image.LANCZOS), dtype=np.uint8)
    return g < 128


def morph_to_target(img: Image.Image, target: float, k: int) -> Image.Image:
    """用 k 核腐蚀/膨胀把墨迹占比往 target 拉一步。"""
    a = np.asarray(img.convert("L").resize((SIZE, SIZE), Image.LANCZOS))
    ink = (a < 128).mean()
    kern = np.ones((k, k), np.uint8)
    # 0=黑：erode→墨变粗，dilate→墨变细
    a2 = cv2.erode(a, kern) if ink < target else cv2.dilate(a, kern)
    return Image.fromarray(a2).convert("RGB")


def case_truth(case_name: str) -> str | None:
    for k, v in KNOWN_TRUTH.items():
        if case_name.endswith("_" + k):
            return v
    return None


def main() -> None:
    setup_font()
    model, names, dev = load_model()
    print(f"模型: {CKPT.name}  设备: {dev}", flush=True)

    sim_dir = ROOT / "data" / "phone_sim"
    rows = list(csv.DictReader((sim_dir / "labels.csv").open(encoding="utf-8-sig")))
    print(f"phone_sim: {len(rows)} 张", flush=True)

    # ---------- 全量预测（用于挑误判 + 统计基准） ----------
    sim_imgs = [Image.open(sim_dir / "clean88" / (Path(r["file"]).stem + ".png")).convert("RGB") for r in rows]
    probs = predict_probs(model, [im.resize((SIZE, SIZE), Image.LANCZOS) for im in sim_imgs], dev)
    preds = probs.argmax(1)
    labels = np.array([int(r["label"]) for r in rows])
    acc_all = float((preds == labels).mean())
    print(f"[基准] phone_sim 全量 {len(rows)} 张，224 直入准确率 = {acc_all:.4f}", flush=True)

    # ---------- 1. 10 张误判对比 ----------
    wrong_idx = [i for i in range(len(rows)) if preds[i] != labels[i]][:10]
    pairs = []
    for i in wrong_idx:
        src = ROOT / rows[i]["source"]                 # data/test_datasets/test/<cls>/<file>.jpg
        rel = Path(rows[i]["source"]).relative_to("data").with_suffix(".png")
        cache_p = ROOT / "data" / "aligned_cache" / rel
        pairs.append((i, src, cache_p, sim_dir / "clean88" / (Path(rows[i]["file"]).stem + ".png")))

    fig, axes = plt.subplots(len(pairs), 2, figsize=(8.6, len(pairs) * 3.0), dpi=120)
    axes = np.atleast_2d(axes)
    rows_json = []
    for k, (i, src, cache_p, clean_p) in enumerate(pairs):
        cache_img = Image.open(cache_p).convert("RGB")
        clean_img = Image.open(clean_p).convert("RGB")
        m1, m2 = mask_of(cache_img), mask_of(clean_img)
        iou = float((m1 & m2).sum() / max(1, (m1 | m2).sum()))
        mt1, mt2 = metrics(cache_img), metrics(clean_img)
        axes[k, 0].imshow(cache_img.resize((224, 224), Image.NEAREST))
        axes[k, 0].set_title(f"aligned_cache（原字）\n黑 {mt1['black']:.2f}", fontsize=8)
        axes[k, 1].imshow(clean_img.resize((224, 224), Image.NEAREST))
        axes[k, 1].set_title(
            f"clean88（洗回）→ 误判 {names[int(preds[i])]} {probs[i].max():.2f}\n"
            f"黑 {mt2['black']:.2f}  IoU {iou:.2f}", fontsize=8)
        for ax in (axes[k, 0], axes[k, 1]):
            ax.axis("off")
        rows_json.append({
            "file": Path(rows[i]["file"]).name,
            "true": names[int(labels[i])],
            "pred": names[int(preds[i])],
            "prob": round(float(probs[i].max()), 4),
            "black_cache": round(mt1["black"], 4),
            "black_clean": round(mt2["black"], 4),
            "iou": round(iou, 4),
        })
    fig.suptitle("同一字：aligned_cache vs phone_sim/clean88（exp6 误判样例）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = ROOT / "outputs" / "cache_vs_clean88.png"
    fig.savefig(out_png, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"[1] 拼图已存 {out_png}", flush=True)

    # ---------- 2. 分布统计 ----------
    rng = np.random.RandomState(SEED)
    cache_dir = ROOT / "data" / "aligned_cache" / "test_datasets" / "test"
    cache_files = sorted(cache_dir.rglob("*.png"))
    pick_cache = [cache_files[i] for i in rng.choice(len(cache_files), N_STAT, replace=False)]
    cache_imgs = [Image.open(p).convert("RGB") for p in pick_cache]
    pick_sim = [sim_imgs[i] for i in rng.choice(len(sim_imgs), N_STAT, replace=False)]

    web = ROOT / "outputs" / "web_test"
    hand = []
    seen = set()
    for d in sorted(web.glob("2026*")):
        orig = next(d.glob("original.*"), None)
        if orig is None:
            continue
        h = hashlib.md5(orig.read_bytes()).hexdigest()
        if h in seen:
            continue
        with Image.open(orig) as im:
            if min(im.size) >= 200:
                hand.append((d, normalize_calligraphy_image(im, size=SIZE, fill=0.88, binarize=True)))
                seen.add(h)
    hand_imgs = [im for _, im in hand]

    groups = {"aligned_cache/test": cache_imgs, "phone_sim/clean88": pick_sim, "真实手写(10)": hand_imgs}
    stat_table = {}
    for gname, imgs in groups.items():
        ms = [metrics(im) for im in imgs]
        stat_table[gname] = {
            key: (round(float(np.mean([m[key] for m in ms])), 4), round(float(np.std([m[key] for m in ms])), 4))
            for key in ("black", "stroke", "edge", "bbox")
        }
    print("\n[2] 分布统计（均值±标准差，128 画布）", flush=True)
    print(f"{'指标':<12s}" + "".join(f"{g:<26s}" for g in stat_table))
    for key, label in (("black", "黑像素占比"), ("stroke", "平均笔画宽"), ("edge", "Canny 边缘比"), ("bbox", "字幅占比")):
        line = f"{label:<12s}"
        for g in stat_table:
            mu, sd = stat_table[g][key]
            line += f"{mu:.4f}±{sd:.4f}".ljust(26)
        print(line, flush=True)

    # ---------- 3. A/B/C 盲测 ----------
    train_black = stat_table["aligned_cache/test"]["black"][0]
    rng2 = np.random.RandomState(SEED + 1)
    idx_sim = rng2.choice(len(sim_imgs), min(N_STAT, len(sim_imgs)), replace=False)
    sim_sub = [sim_imgs[i] for i in idx_sim]
    sim_y = labels[idx_sim]

    def eval_group(imgs, y, transform_fn, tag):
        ts = [transform_fn(im) for im in imgs]
        p = predict_probs(model, ts, dev)
        acc = float((p.argmax(1) == y).mean())
        print(f"  {tag:<34s} acc={acc:.4f}", flush=True)
        return acc, p

    def to224(im):
        return im.resize((SIZE, SIZE), Image.LANCZOS)

    def to64_back(im):
        small = im.resize((64, 64), Image.LANCZOS)
        return small.resize((SIZE, SIZE), Image.LANCZOS)

    def best_morph(im):
        best, best_gap = im, 9e9
        for k in (2, 3):
            for _ in range(2):
                m = morph_to_target(im, train_black, k)
                gap = abs(metrics(m)["black"] - train_black)
                if gap < best_gap:
                    best, best_gap = m, gap
        return best

    print("\n[3] 零成本盲测（phone_sim 抽 %d 张）" % len(sim_sub), flush=True)
    res = {}
    res["A_224"], pa = eval_group(sim_sub, sim_y, to224, "A 224 直入（基准）")
    res["B_64"], pb = eval_group(sim_sub, sim_y, to64_back, "B 先降 64 再拉回 224")
    res["C_morph"], pc = eval_group(sim_sub, sim_y, best_morph, "C 形态学纠墨量到训练均值")

    # 手写 10 张的三组预测
    hand_y = [case_truth(d.name) for d, _ in hand]
    print("\n  手写 10 张在 A/B/C 下的 Top-1 变化:", flush=True)
    hand_detail = []
    pa_h = predict_probs(model, [to224(im) for im in hand_imgs], dev)
    pb_h = predict_probs(model, [to64_back(im) for im in hand_imgs], dev)
    pc_h = predict_probs(model, [best_morph(im) for im in hand_imgs], dev)
    for k, (d, im) in enumerate(hand):
        truth = hand_y[k]
        def fmt(p):
            j = int(p.argmax())
            return f"{names[j]}:{p[j]:.2f}"
        rank = "—"
        if truth:
            order = np.argsort(-pa_h[k])
            rank = f"第{int(np.where(order == names.index(truth))[0][0]) + 1}名"
        line = (f"  {d.name.split('_')[-1]}  真值={truth or '未知'}  "
                f"A={fmt(pa_h[k])}  B={fmt(pb_h[k])}  C={fmt(pc_h[k])}  {rank}")
        print(line, flush=True)
        hand_detail.append({
            "case": d.name, "truth": truth, "rank_under_A": rank,
            "A": fmt(pa_h[k]), "B": fmt(pb_h[k]), "C": fmt(pc_h[k]),
            "A_top3": [{"name": names[int(j)], "prob": round(float(pa_h[k][j]), 4)}
                       for j in np.argsort(-pa_h[k])[:3]],
        })

    # ---------- 汇总 ----------
    payload = {
        "exp": "exp6_convnext_224",
        "phone_sim_full_acc": round(acc_all, 4),
        "wrong_examples": rows_json,
        "stats": stat_table,
        "train_black_mean": round(train_black, 4),
        "blind_test": {"A_224": round(res["A_224"], 4), "B_64": round(res["B_64"], 4),
                       "C_morph": round(res["C_morph"], 4)},
        "handwritten": hand_detail,
    }
    out_json = ROOT / "outputs" / "diagnose_60pt_drop.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
