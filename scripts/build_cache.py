# -*- coding: utf-8 -*-
"""构建离线对齐缓存：把每张原图清洗成「0.88 字幅 + 二值化」的 128×128 干净图。

只缓存**清洗对齐后的干净原图**，不缓存任何增强结果（增强在 DataLoader 里每轮随机现做）。

输出目录结构（与原 data/ 一一对应）：
    data/aligned_cache/train_datasets/train/<cls>/<stem>.png
    data/aligned_cache/test_datasets/test/<cls>/<stem>.png
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
CACHE = DATA / "aligned_cache"
SIZE = 128
FILL = 0.88


def _process(chunk: list[tuple[str, str]]) -> tuple[int, int]:
    from utils.preprocess import normalize_calligraphy_image

    ok = fail = 0
    for src, dst in chunk:
        try:
            with Image.open(src) as im:
                out = normalize_calligraphy_image(im, size=SIZE, fill=FILL, binarize=True)
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            out.save(dst, format="PNG", optimize=False)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            fail += 1
            if fail <= 5:
                print(f"[skip] {src}: {type(exc).__name__}: {exc}", flush=True)
    return ok, fail


def collect() -> list[tuple[str, str]]:
    pairs = []
    for root_name in ("train_datasets", "test_datasets"):
        for p in sorted((DATA / root_name).rglob("*.jpg")):
            rel = p.relative_to(DATA)
            pairs.append((str(p), str((CACHE / rel).with_suffix(".png"))))
    return pairs


def main() -> None:
    pairs = collect()
    print(f"共 {len(pairs)} 张待清洗", flush=True)
    n_workers = 8
    chunks = [pairs[i::n_workers * 4] for i in range(n_workers * 4)]
    chunks = [c for c in chunks if c]
    t0 = time.perf_counter()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (o, f) in enumerate(ex.map(_process, chunks), 1):
            ok += o
            fail += f
            print(f"  chunk {i}/{len(chunks)}  ok={ok} fail={fail}  {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"完成: ok={ok} fail={fail} 用时 {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
