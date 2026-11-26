# -*- coding: utf-8 -*-
"""通宵实验队列：干跑 → 正式训练 → 三套卷子 → 病理体检，全程容错。

规则：
* 每组正式跑之前先做 3 个 batch 的干跑（含前反向），不过则记录原因并跳下一组
* 任何一组训练报错（含 OOM）都写日志、清显存、继续下一组，绝不中断队列
* 第 6 组只在「开跑时间早于 03:00」时启动
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

PY = r"D:\anaconda3\envs\CNNN\python.exe"
LOG = ROOT / "outputs" / "overnight_run.log"
FAIL = ROOT / "outputs" / "overnight_failures.log"
EVAL_DIR = ROOT / "outputs" / "evals"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    dict(name="exp2_domain_aligned", arch="resnet18", size=128, epochs=25, bs=64, lr=1e-4,
         flags=["--domain_align"]),
    dict(name="exp2b_coverage", arch="resnet18", size=128, epochs=25, bs=64, lr=1e-4,
         flags=["--domain_align", "--coverage_rand"]),
    dict(name="exp3_hflip_bad", arch="resnet18", size=128, epochs=25, bs=64, lr=1e-4,
         flags=["--domain_align", "--allow_hflip"]),
    dict(name="exp4_resnet18_cbam", arch="resnet18_cbam", size=128, epochs=30, bs=64, lr=1e-4,
         flags=["--domain_align"]),
    dict(name="exp5_convnext_tiny", arch="convnext_tiny", size=128, epochs=30, bs=32, lr=2e-4,
         flags=["--domain_align", "--class_weight"]),
    dict(name="exp6_convnext_224", arch="convnext_tiny", size=224, epochs=30, bs=16, lr=2e-4,
         flags=["--domain_align", "--class_weight"], before_hour=3),
]


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def fail(msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print("[FAIL] " + line, flush=True)
    with FAIL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def dry_run(cfg: dict) -> tuple[bool, str]:
    """3 个 batch 干跑：数据 → 前向 → 反向 → 优化器 step。"""
    try:
        import torch
        from models.networks import build_model
        from utils.dataset import build_dataloaders, build_train_transforms

        dev = torch.device("cuda")
        dl = build_dataloaders(img_size=cfg["size"], batch_size=cfg["bs"], num_workers=0,
                               domain_align="--domain_align" in cfg["flags"],
                               coverage_rand="--coverage_rand" in cfg["flags"],
                               allow_hflip="--allow_hflip" in cfg["flags"])
        model = build_model(cfg["arch"], num_classes=20, pretrained=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        crit = torch.nn.CrossEntropyLoss()
        model.train()
        it = iter(dl["train"])
        for i in range(3):
            x, y = next(it)
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        vx, vy = next(iter(dl["val"]))
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            model(vx.to(dev))
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        del model, opt, dl
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return True, f"dry-run ok, 3 batch, 峰值 {peak:.0f} MB"
    except Exception as exc:  # noqa: BLE001
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1200:]}"


def run(cmd: list[str], timeout: int = 6 * 3600) -> tuple[int, str]:
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main() -> None:
    LOG.write_text("", encoding="utf-8")
    FAIL.write_text("", encoding="utf-8")
    log("==== 通宵队列开始 ====")

    for cfg in CONFIGS:
        name = cfg["name"]
        if cfg.get("before_hour") is not None:
            now_h = datetime.now().hour
            if now_h >= cfg["before_hour"] and now_h < 12:
                log(f"[skip] {name}: 已过 {cfg['before_hour']}:00（当前 {datetime.now():%H:%M}）")
                fail(f"{name}: 跳过（启动时间超过 {cfg['before_hour']}:00）")
                continue

        log(f"---- {name} 干跑 ----")
        ok, msg = dry_run(cfg)
        log(f"{name} 干跑: {msg.splitlines()[0]}")
        if not ok:
            fail(f"{name}: 干跑失败\n{msg}")
            continue

        log(f"---- {name} 正式训练 ----")
        cmd = [PY, str(ROOT / "train.py"), "--exp_name", name, "--model", cfg["arch"],
               "--img_size", str(cfg["size"]), "--epochs", str(cfg["epochs"]),
               "--batch_size", str(cfg["bs"]), "--lr", str(cfg["lr"])] + cfg["flags"]
        t0 = time.time()
        try:
            code, out = run(cmd)
        except subprocess.TimeoutExpired:
            fail(f"{name}: 训练超时（>6h）")
            continue
        (ROOT / "outputs" / f"train_{name}.log").write_text(out, encoding="utf-8")
        dur = (time.time() - t0) / 60
        if code != 0:
            tail = "\n".join(out.strip().splitlines()[-25:])
            fail(f"{name}: 训练退出码 {code}（{dur:.0f} min）\n{tail}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            continue

        ckpt = ROOT / "checkpoints" / name / "best.pth"
        if not ckpt.exists():
            fail(f"{name}: 训练结束但没有 best.pth")
            continue
        log(f"{name} 训练完成 {dur:.0f} min → {ckpt}")

        # ---- 三套卷子 ----
        eval_out = EVAL_DIR / f"eval_{name}.json"
        ecmd = [PY, str(ROOT / "scripts" / "eval_model.py"), "--exp", name,
                "--arch", cfg["arch"], "--preprocess", "domain",
                "--fill", "0.88", "--binarize", "1", "--img_size", str(cfg["size"]),
                "--out", str(eval_out)]
        code, out = run(ecmd, timeout=3600)
        (ROOT / "outputs" / f"eval_{name}.log").write_text(out, encoding="utf-8")
        if code != 0:
            fail(f"{name}: 评测失败\n" + "\n".join(out.strip().splitlines()[-20:]))
        else:
            log(f"{name} 评测完成 → {eval_out}")

        # ---- 病理体检 ----
        pcmd = [PY, str(ROOT / "scripts" / "check_model_pathology.py"), "--exp", name,
                "--arch", cfg["arch"], "--fill", "0.88", "--binarize", "1",
                "--img_size", str(cfg["size"]),
                "--out", str(ROOT / "outputs" / f"pathology_{name}.json")]
        code, out = run(pcmd, timeout=1200)
        (ROOT / "outputs" / f"pathology_{name}.log").write_text(out, encoding="utf-8")
        log(f"{name} 病理体检: {'ok' if code == 0 else '失败'}")

    log("==== 队列结束 ====")


if __name__ == "__main__":
    main()
