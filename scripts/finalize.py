# -*- coding: utf-8 -*-
"""收尾：按官方分数挑最佳模型 → 混淆矩阵 + Grad-CAM → 生成 overnight_report.md。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"G:\深度学习期末报告")
PY = sys.executable
EVAL_DIR = ROOT / "outputs" / "evals"


def main() -> None:
    evals = {}
    for p in EVAL_DIR.glob("eval_*.json"):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
            evals[d["exp"]] = d
        except Exception:  # noqa: BLE001
            pass
    if not evals:
        print("没有可用评测结果")
        return
    best = max(evals.values(), key=lambda d: d["official"]["top1"])
    name = best["exp"]
    cfg_p = ROOT / "outputs" / name / "config.json"
    img_size = 128
    if cfg_p.exists():
        try:
            img_size = int(json.loads(cfg_p.read_text(encoding="utf-8-sig")).get("img_size", 128))
        except Exception:  # noqa: BLE001
            pass
    print(f"最佳模型: {name}  官方 top1={best['official']['top1']:.4f}  分辨率={img_size}")

    for script, extra in (("make_confusion.py", []), ("make_gradcam.py", [])):
        cmd = [PY, str(ROOT / "scripts" / script), "--exp", name, "--arch", best["arch"],
               "--img_size", str(img_size)] + extra
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        print(f"  {script}: {p.returncode}  {(p.stdout or '').strip().splitlines()[-1:] or ''}")
        (ROOT / "outputs" / f"{script}.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")

    p = subprocess.run([PY, str(ROOT / "scripts" / "make_report.py")], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print((p.stdout or "").strip(), (p.stderr or "")[-500:])


if __name__ == "__main__":
    main()
