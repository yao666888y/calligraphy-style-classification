# -*- coding: utf-8 -*-
"""把模拟手机照片按「域对齐」口径（0.88 字幅 + 二值化）预清洗一次，供所有域对齐模型复用。"""
from __future__ import annotations

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(r"G:\深度学习期末报告")
sys.path.insert(0, str(ROOT))

SIM = ROOT / "data" / "phone_sim"
OUT = SIM / "clean88"


def _one(p: str) -> bool:
    from utils.preprocess import normalize_calligraphy_image

    try:
        with Image.open(p) as im:
            out = normalize_calligraphy_image(im, size=128, fill=0.88, binarize=True)
        out.save(OUT / (Path(p).stem + ".png"))
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((SIM / "raw").glob("*.png"))
    print(f"{len(files)} 张待清洗", flush=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        ok = sum(ex.map(_one, [str(f) for f in files], chunksize=32))
    print(f"完成 {ok}/{len(files)} → {OUT}")


if __name__ == "__main__":
    main()
