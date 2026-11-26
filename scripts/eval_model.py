# -*- coding: utf-8 -*-
"""单个模型的「三套卷子」评测，结果写 JSON。

    python scripts/eval_model.py --exp exp2_domain_aligned --arch resnet18 \
        --preprocess domain --fill 0.88 --out outputs/eval_exp2.json

三套卷子：
  A. 官方对齐测试集（data/aligned_cache 的 test 部分，0.88 二值）
  B. 模拟手机照片（data/phone_sim/raw，先走该模型的拍照清洗流水线）
  C. 手写样本（outputs/web_test 里用户上传的真实照片，>=200px，走同一清洗入口）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import build_dataloaders, build_eval_transforms, denormalize  # noqa: E402
from utils.preprocess import normalize_calligraphy_image  # noqa: E402


def predict_batch(model, tensors, device):
    preds = []
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for i in range(0, len(tensors), 64):
            chunk = torch.stack(tensors[i:i + 64]).to(device)
            preds.append(model(chunk).argmax(1).cpu())
    return torch.cat(preds).numpy()


def clean_for(im: Image.Image, args, size: int = 128) -> Image.Image:
    rgb = im.convert("RGB")
    if max(rgb.size) > 2000:
        s = 2000 / max(rgb.size)
        rgb = rgb.resize((max(1, int(rgb.width * s)), max(1, int(rgb.height * s))), Image.LANCZOS)
    return normalize_calligraphy_image(
        rgb, size=size, fill=args.fill, binarize=args.binarize,
        mask_background="auto" if args.preprocess == "web_v1" else True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--preprocess", default="domain", choices=["domain", "web_v1"])
    ap.add_argument("--fill", type=float, default=0.88)
    ap.add_argument("--binarize", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.binarize = bool(args.binarize)

    ckpt_path = Path(args.ckpt) if args.ckpt else ROOT / "checkpoints" / args.exp / "best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    model = build_model(args.arch, num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    tf = build_eval_transforms(args.img_size, "imagenet")
    result = {"exp": args.exp, "arch": args.arch, "fill": args.fill,
              "binarize": args.binarize, "preprocess": args.preprocess,
              "best_epoch": ck.get("epoch"), "val_macro_f1": ck.get("val_macro_f1")}

    # ---------- A. 官方对齐测试集 ----------
    dl = build_dataloaders(img_size=args.img_size, batch_size=64, num_workers=0,
                           domain_align=True, domain_fill=0.88)
    preds, trues = [], []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        for x, y in dl["test"]:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu())
            trues.append(y)
    p, t = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    top1 = float((p == t).mean())
    result["official"] = {"n": int(len(t)), "top1": round(top1, 4),
                          "macro_f1": round(float(f1_score(t, p, average="macro", zero_division=0)), 4)}
    print(f"[A] 官方对齐测试集 n={len(t)} top1={top1:.4f}", flush=True)

    # ---------- B. 模拟手机照片 ----------
    sim_dir = ROOT / "data" / "phone_sim"
    rows = list(csv.DictReader((sim_dir / "labels.csv").open(encoding="utf-8-sig")))
    use_preclean = abs(args.fill - 0.88) < 1e-6 and args.binarize
    tensors, labels = [], []
    for r in rows:
        if use_preclean:
            with Image.open(sim_dir / "clean88" / (Path(r["file"]).stem + ".png")) as im:
                tensors.append(tf(im.convert("RGB")))
        else:
            with Image.open(sim_dir / "raw" / r["file"]) as im:
                tensors.append(tf(clean_for(im, args, args.img_size)))
        labels.append(int(r["label"]))
    ps = predict_batch(model, tensors, device)
    ts = np.array(labels)
    result["phone_sim"] = {"n": int(len(ts)), "top1": round(float((ps == ts).mean()), 4),
                           "macro_f1": round(float(f1_score(ts, ps, average="macro", zero_division=0)), 4)}
    print(f"[B] 模拟手机 n={len(ts)} top1={(ps == ts).mean():.4f}", flush=True)

    # ---------- C. 手写样本 ----------
    import hashlib

    web = ROOT / "outputs" / "web_test"
    samples, seen = [], set()
    for d in sorted(web.glob("2026*")):
        orig = next(d.glob("original.*"), None)
        if orig is None:
            continue
        raw = orig.read_bytes()
        h = hashlib.md5(raw).hexdigest()
        if h in seen:                 # 同一张图重复上传只考一次
            continue
        with Image.open(orig) as im:
            if min(im.size) >= 200:   # 排除 64px 数据集图，只要真实照片
                samples.append(orig)
                seen.add(h)
    used, details = [], []
    for p_ in samples:
        with Image.open(p_) as im:
            t_ = tf(clean_for(im, args, args.img_size)).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            prob = torch.softmax(model(t_).float(), 1)[0].cpu()
        v, i = prob.topk(3)
        details.append({
            "file": str(p_.relative_to(ROOT)),
            "top3": [{"name": classes[int(j)], "prob": round(float(s), 4)} for s, j in zip(v, i)],
        })
        used.append(str(p_.relative_to(ROOT)))
    result["handwritten"] = {"n": len(details), "source": str(web), "details": details}
    print(f"[C] 手写样本 n={len(details)}", flush=True)

    out = Path(args.out) if args.out else ROOT / "outputs" / f"eval_{args.exp}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
