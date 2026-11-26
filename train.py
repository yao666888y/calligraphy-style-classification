# -*- coding: utf-8 -*-
"""训练主脚本（Step 3）。

用法示例
--------
    python train.py --exp_name smoke_test --model resnet18 --epochs 1
    python train.py --exp_name r18_smallstem --small_stem --epochs 30 --class_weight

产出
----
    checkpoints/<exp_name>/best.pth        验证集 Macro-F1 最高的权重
    outputs/<exp_name>/history.csv          逐 epoch 日志（追加写，utf-8-sig）
    outputs/<exp_name>/curves.png           Loss 与 Acc/F1 曲线
    outputs/<exp_name>/config.json          本次运行的完整配置
    outputs/experiments_summary.csv         每组实验的测试集最终成绩（追加写）

训练结束后会用 best.pth 在**独立 test 集**上评估一次，写入总表；
早停：验证集 Macro-F1 连续 --patience（默认 6）轮无提升即结束。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.networks import build_model, describe  # noqa: E402
from utils.dataset import build_dataloaders  # noqa: E402

HISTORY_COLS = [
    "run_id", "epoch", "lr", "train_loss", "train_acc",
    "val_loss", "val_top1", "val_top3", "val_macro_f1",
    "epoch_time_s", "peak_mem_mb",
]

SUMMARY_COLS = [
    "exp_name", "model", "allow_hflip", "class_weight",
    "params(M)", "test_top1", "test_top3", "test_macro_f1",
]


# --------------------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True


def setup_font() -> None:
    cands = ["Microsoft YaHei", "SimHei", "SimSun"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    pick = next((c for c in cands if c in have), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [pick] + cands
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="中国历代书法名家分类 - 训练")
    p.add_argument("--exp_name", type=str, default="exp", help="实验名，决定输出目录")
    p.add_argument("--model", type=str, default="resnet18", help="网络名（当前支持 resnet18）")
    p.add_argument("--img_size", type=int, default=128, help="输入分辨率")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW 初始学习率")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--small_stem", action="store_true", help="3x3/s1 stem + 去掉首层 maxpool")
    p.add_argument("--allow_hflip", action="store_true", help="开启水平翻转（仅反面消融实验用）")
    p.add_argument("--class_weight", action="store_true", help="按类别逆频率加权交叉熵")
    p.add_argument("--domain_align", action="store_true",
                   help="域对齐：全流程使用共享归一化（紧致裁切+88%%字幅+二值化），"
                        "并追加随机留白/形态学增强")
    p.add_argument("--coverage_rand", action="store_true", help="域对齐时追加墨迹覆盖率纠正（>0.28 或 <0.10 才往区间内拉）")
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="不加载 ImageNet 预训练")
    p.add_argument("--max_batches", type=int, default=0, help="调试用：每个 epoch 只跑前 N 个 batch（0=不限）")
    p.add_argument("--patience", type=int, default=6, help="早停耐心值：验证 Macro-F1 连续 N 轮无提升即停（0=关闭）")
    p.add_argument("--replot", action="store_true", help="不训练，仅用已有 history.csv 重画 curves.png 后退出")
    return p.parse_args()


def build_class_weights(counts: list[int], device: torch.device) -> torch.Tensor:
    """逆频率权重：w_c = N / (C * n_c)，均值归一到 1。"""
    c = torch.tensor(counts, dtype=torch.float32)
    w = c.sum() / (len(c) * c.clamp(min=1))
    return (w / w.mean()).to(device)


def topk_correct(logits: torch.Tensor, target: torch.Tensor, k: int) -> int:
    _, pred = logits.topk(k, dim=1)
    return int((pred == target.view(-1, 1)).any(dim=1).sum())


# --------------------------------------------------------------------------------------
# 单轮训练 / 验证
# --------------------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, epochs, max_batches=0):
    model.train()
    total_loss, total_correct, total_n = 0.0, 0, 0
    amp = scaler.is_enabled()
    desc = f"train {epoch}/{epochs}"
    for i, (images, labels) in enumerate(tqdm(loader, desc=desc, ncols=100, leave=False)):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_n += labels.size(0)

        if i % 200 == 0:
            tqdm.write(
                f"    [{desc}] iter {i}/{len(loader)}  "
                f"loss={total_loss / max(1, total_n):.4f}  acc={total_correct / max(1, total_n):.4f}"
            )
    n = max(1, total_n)
    return total_loss / n, total_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_n = 0
    top1 = top3 = 0
    all_pred, all_true = [], []

    for i, (images, labels) in enumerate(tqdm(loader, desc="val", ncols=100, leave=False)):
        if max_batches and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)
        top1 += (logits.argmax(1) == labels).sum().item()
        top3 += topk_correct(logits, labels, 3)
        all_pred.append(logits.argmax(1).cpu())
        all_true.append(labels.cpu())

    n = max(1, total_n)
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()
    macro_f1 = float(f1_score(true, pred, average="macro", zero_division=0))
    return {
        "loss": total_loss / n,
        "top1": top1 / n,
        "top3": top3 / n,
        "macro_f1": macro_f1,
    }


# --------------------------------------------------------------------------------------
# 记录与绘图
# --------------------------------------------------------------------------------------
def append_history(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=HISTORY_COLS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in HISTORY_COLS})


def append_summary(path: Path, row: dict) -> None:
    """把本次实验的测试集成绩追加到总表。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k) for k in SUMMARY_COLS})


def plot_curves(csv_path: Path, out_png: Path, run_id: str, best_epoch: int) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df[df["run_id"] == run_id].sort_values("epoch")
    if df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150)
    x_max = max(1.5, float(df["epoch"].max()) + 0.5)
    x_min = 0.5

    ax = axes[0]
    ax.plot(df["epoch"], df["train_loss"], "o-", label="训练 Loss", lw=2)
    ax.plot(df["epoch"], df["val_loss"], "s-", label="验证 Loss", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("损失曲线")
    ax.set_xlim(x_min, x_max)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(df["epoch"], df["train_acc"], "o-", label="训练 Acc", lw=2)
    ax.plot(df["epoch"], df["val_top1"], "s-", label="验证 Top-1", lw=2)
    ax.plot(df["epoch"], df["val_top3"], "^-", label="验证 Top-3", lw=2)
    ax.plot(df["epoch"], df["val_macro_f1"], "d-", label="验证 Macro-F1", lw=2)
    if best_epoch in set(df["epoch"]):
        row = df[df["epoch"] == best_epoch].iloc[0]
        ax.axvline(best_epoch, ls=":", c="gray", alpha=0.7)
        ax.scatter([best_epoch], [row["val_macro_f1"]], s=120, facecolors="none",
                   edgecolors="red", zorder=5, label=f"best (ep{best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy / F1")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(x_min, x_max)
    ax.set_title("准确率与 F1 曲线")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(f"训练曲线 · {run_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    setup_font()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    run_id = f"{args.exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ROOT / "outputs" / args.exp_name
    ckpt_dir = ROOT / "checkpoints" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"

    if args.replot:            # 只重画曲线，不重新训练
        import pandas as pd

        df = pd.read_csv(history_path, encoding="utf-8-sig")
        rid = df["run_id"].iloc[-1]
        sub = df[df["run_id"] == rid]
        best_ep = int(sub.loc[sub["val_macro_f1"].idxmax(), "epoch"])
        plot_curves(history_path, out_dir / "curves.png", rid, best_ep)
        print(f"[replot] {rid}: {len(sub)} epoch, best epoch={best_ep} → {out_dir / 'curves.png'}")
        return 0

    print("=" * 78)
    print(f"实验: {args.exp_name}   run_id: {run_id}")
    print("=" * 78)
    print(f"设备: {device}  ({torch.cuda.get_device_name(0) if use_amp else 'CPU'})")
    print(f"AMP : {use_amp}   模型: {args.model}   img_size: {args.img_size}   small_stem: {args.small_stem}")

    # ---------- 数据 ----------
    t0 = time.perf_counter()
    dl = build_dataloaders(
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        allow_hflip=args.allow_hflip,
        domain_align=args.domain_align,
        coverage_rand=args.coverage_rand,
    )
    s = dl["stats"]
    classes = dl["classes"]
    print(f"数据: train={s['n_train']} val={s['n_val']} test(不参与调参)={s['n_test']}  "
          f"batch={args.batch_size}  workers={args.num_workers}  ({time.perf_counter() - t0:.1f}s)")
    if args.domain_align:
        print(f"域对齐: 开启（共享归一化 fill={s['domain_fill']}, 二值化, 随机留白 78%~94%, 形态学 p=0.3）")

    # ---------- 模型 ----------
    model = build_model(
        args.model,
        num_classes=s["n_classes"],
        small_stem=args.small_stem,
        pretrained=not args.no_pretrained,
    ).to(device)
    print(f"模型: {describe(model)}")

    # ---------- 损失 ----------
    cw = None
    if args.class_weight:
        cw = build_class_weights(s["train_per_class"], device)
        print(f"类别权重(逆频率): min={cw.min():.3f} max={cw.max():.3f} "
              f"... 对应 {classes[int(cw.argmax())]} 权重最大")
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    print(f"损失: CrossEntropyLoss(label_smoothing={args.label_smoothing}, class_weight={'on' if cw is not None else 'off'})")

    # ---------- 优化器 / 调度 / AMP ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"优化器: AdamW(lr={args.lr}, wd={args.weight_decay})  调度: CosineAnnealingLR(T_max={args.epochs})")

    # 配置落盘
    (out_dir / "config.json").write_text(
        json.dumps({"run_id": run_id, **vars(args), "stats": s}, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    # ---------- 训练 ----------
    history: list[dict] = []
    best_f1, best_epoch = -1.0, -1
    no_improve = 0
    stopped_early = False
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]

        tr_loss, tr_acc = train_one_epoch(
            model, dl["train"], criterion, optimizer, scaler, device, epoch, args.epochs, args.max_batches
        )
        va = evaluate(model, dl["val"], criterion, device, args.max_batches)
        scheduler.step()

        ep_time = time.perf_counter() - t_ep
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if use_amp else 0.0

        row = {
            "run_id": run_id,
            "epoch": epoch,
            "lr": f"{lr_now:.6g}",
            "train_loss": f"{tr_loss:.4f}",
            "train_acc": f"{tr_acc:.4f}",
            "val_loss": f"{va['loss']:.4f}",
            "val_top1": f"{va['top1']:.4f}",
            "val_top3": f"{va['top3']:.4f}",
            "val_macro_f1": f"{va['macro_f1']:.4f}",
            "epoch_time_s": f"{ep_time:.1f}",
            "peak_mem_mb": f"{peak_mb:.0f}",
        }
        history.append(row)
        append_history(history_path, [row])   # 每个 epoch 立即追加，避免中途崩溃丢日志

        print(f"  epoch {epoch}/{args.epochs}  {ep_time:.1f}s  lr={lr_now:.2e}")
        print(f"    train  loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"    val    loss={va['loss']:.4f}  top1={va['top1']:.4f}  top3={va['top3']:.4f}  macroF1={va['macro_f1']:.4f}")
        print(f"    显存峰值 {peak_mb:.0f} MB" if use_amp else "    CPU 训练")

        if va["macro_f1"] > best_f1:
            best_f1, best_epoch = va["macro_f1"], epoch
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": va["macro_f1"],
                    "val_top1": va["top1"],
                    "classes": classes,
                    "args": vars(args),
                    "run_id": run_id,
                },
                ckpt_dir / "best.pth",
            )
            print(f"    [best] Macro-F1={best_f1:.4f} → checkpoints/{args.exp_name}/best.pth")
        else:
            no_improve += 1
            print(f"    [no-improve] {no_improve}/{args.patience} (best={best_f1:.4f} @ep{best_epoch})")
            if args.patience > 0 and no_improve >= args.patience:
                stopped_early = True
                print(f"    [early stop] 验证 Macro-F1 连续 {args.patience} 轮无提升，提前结束")
                break

    total = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"完成: {len(history)} epoch 用时 {total / 60:.1f} min  "
          f"（{total / max(1, len(history)):.1f}s/epoch）")
    print(f"最佳: epoch {best_epoch}, val Macro-F1={best_f1:.4f}"
          + ("（触发早停）" if stopped_early else ""))
    if use_amp:
        print(f"峰值显存: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB "
              f"/ {torch.cuda.get_device_properties(0).total_memory / 1024 ** 2:.0f} MB")
    print(f"权重: {ckpt_dir / 'best.pth'}")
    print(f"日志: {history_path}")

    plot_curves(history_path, out_dir / "curves.png", run_id, best_epoch)
    print(f"曲线: {out_dir / 'curves.png'}")

    # ---------- 用 best.pth 在独立 test 集上评估 ----------
    print("\n" + "-" * 78)
    print("测试集最终评估（best.pth，test 全程未参与调参）")
    ckpt = torch.load(ckpt_dir / "best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    # 用独立的 num_workers=0 loader 做最终评估：
    # 训练末期再 spawn 第 4 组 worker 在 Windows 上会卡死（实测阻塞十几分钟）
    from torch.utils.data import DataLoader

    test_loader = DataLoader(dl["test_ds"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = evaluate(model, test_loader, criterion, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  test  loss={te['loss']:.4f}  top1={te['top1']:.4f}  "
          f"top3={te['top3']:.4f}  macroF1={te['macro_f1']:.4f}")

    summary_row = {
        "exp_name": args.exp_name,
        "model": args.model,
        "allow_hflip": str(bool(args.allow_hflip)),
        "class_weight": str(bool(args.class_weight)),
        "params(M)": f"{n_params:.2f}",
        "test_top1": f"{te['top1']:.4f}",
        "test_top3": f"{te['top3']:.4f}",
        "test_macro_f1": f"{te['macro_f1']:.4f}",
    }
    summary_path = ROOT / "outputs" / "experiments_summary.csv"
    append_summary(summary_path, summary_row)
    print(f"  已追加到 {summary_path}")

    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())

