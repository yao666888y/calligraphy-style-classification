# -*- coding: utf-8 -*-
"""装包后的功能冒烟测试：不只看 import 成功，而是真跑一遍。

覆盖: torch CUDA / torchvision transforms / timm 建模前反向 / cv2 中文路径读写 /
      sklearn 训练 / pandas + openpyxl 读 Summary.csv / grad-cam API
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

DATA = Path(r"G:\深度学习期末报告\data")
TMP = Path(r"G:\深度学习期末报告\outputs\_smoke")
TMP.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            msg = fn() or ""
            results.append((name, True, str(msg)))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
        return fn

    return deco


@check("1. torch CUDA 张量运算")
def _():
    import torch

    assert torch.__version__ == "2.11.0+cu130", f"torch 版本被改动: {torch.__version__}"
    assert torch.cuda.is_available(), "CUDA 不可用"
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    s = (a @ b).sum().item()
    torch.cuda.synchronize()
    return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, sum={s:.1f}"


@check("2. timm 建模型 + GPU 前向/反向")
def _():
    import timm
    import torch

    m = timm.create_model("resnet18", pretrained=False, num_classes=20).cuda()
    x = torch.randn(4, 3, 64, 64, device="cuda")
    out = m(x)
    out.sum().backward()
    return f"timm {timm.__version__}, resnet18 out={tuple(out.shape)}, 参数可反传"


@check("3. torchvision transforms 管线")
def _():
    import torchvision
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
    img = Image.new("RGB", (64, 64), "white")
    t = tf(img)
    return f"torchvision {torchvision.__version__}, tensor {tuple(t.shape)}"


@check("4. cv2 中文路径 读→写→再读")
def _():
    import cv2

    src = next((DATA / "train_datasets" / "train" / "wxz").glob("*.jpg"))
    buf = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (64, 64, 3), f"imdecode 结果异常: {None if img is None else img.shape}"
    dst = TMP / "中文路径_测试.jpg"
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    enc.tofile(str(dst))  # 不用 cv2.imwrite
    from PIL import Image

    with Image.open(dst) as im:
        assert im.size == (64, 64)
    return f"cv2 {cv2.__version__}, imdecode {img.shape}, imencode→tofile 落盘 {dst.stat().st_size}B 并回读成功"


@check("5. scikit-learn 训练")
def _():
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = np.random.RandomState(0).randn(200, 8)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
    return f"sklearn {sklearn.__version__}, test acc={clf.score(Xte, yte):.3f}"


@check("6. pandas + openpyxl 读 Summary")
def _():
    import pandas as pd

    cands = [
        p for p in DATA.iterdir()
        if p.is_file() and p.stem.lower() == "summary" and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
    ]
    assert cands, "data 目录下找不到 Summary.csv / Summary.xlsx"
    path = cands[0]
    if path.suffix.lower() == ".csv" and path.read_bytes()[:2] != b"PK":
        df = pd.read_csv(path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(path, engine="openpyxl")
    assert df.shape[0] == 20, df.shape
    return f"pandas {pd.__version__}, 读 {path.name}: {df.shape[0]} 行 × {df.shape[1]} 列, 列名={list(df.columns)}"


@check("7. grad-cam(CAM) API")
def _():
    import pytorch_grad_cam
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM

    return f"pytorch_grad_cam 导入成功, 可用: GradCAM/GradCAMPlusPlus/ScoreCAM, 路径 {Path(pytorch_grad_cam.__file__).parent.name}"


@check("8. tqdm 进度条对象")
def _():
    import tqdm

    total = sum(1 for _ in tqdm.tqdm(range(100), disable=True))
    return f"tqdm {tqdm.__version__}, 遍历 {total} 项"


print("=" * 74)
print("装包后功能冒烟测试")
print("=" * 74)
n_ok = 0
for name, ok, msg in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         {msg}")
    n_ok += ok
print()
print(f"结果: {n_ok}/{len(results)} 通过")
if n_ok != len(results):
    print("[FAIL] 有功能未通过，见上方 traceback")
    sys.exit(1)
print("[OK] 全部功能可用")
