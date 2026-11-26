# -*- coding: utf-8 -*-
"""Step 1-1/1-2: 检查 CNNN 环境的 Python / PyTorch / CUDA / GPU 与依赖清单。"""
import importlib
import json
import platform
import sys

REPORT = {}

# ---------- Python ----------
REPORT["python"] = {
    "version": sys.version.split()[0],
    "full": sys.version.replace("\n", " "),
    "executable": sys.executable,
    "arch": platform.architecture()[0],
    "platform": platform.platform(),
}

# ---------- PyTorch / CUDA ----------
try:
    import torch

    torch_info = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "devices": [],
    }
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        torch_info["devices"].append(
            {
                "index": i,
                "name": p.name,
                "total_memory_GiB": round(p.total_memory / 1024 ** 3, 2),
                "compute_capability": f"{p.major}.{p.minor}",
                "multi_processor_count": p.multi_processor_count,
            }
        )
    if torch.cuda.is_available():
        torch_info["current_device"] = torch.cuda.current_device()
        torch_info["matmul_smoke_test"] = True
        try:
            a = torch.randn(256, 256, device="cuda")
            b = torch.randn(256, 256, device="cuda")
            _ = (a @ b).sum().item()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            torch_info["matmul_smoke_test"] = False
            torch_info["matmul_error"] = f"{type(exc).__name__}: {exc}"
    REPORT["torch"] = torch_info
except Exception as exc:  # noqa: BLE001
    REPORT["torch"] = {"error": f"{type(exc).__name__}: {exc}"}

# ---------- 依赖清单 ----------
DEPS = [
    ("torch", "torch", "必需"),
    ("torchvision", "torchvision", "必需"),
    ("timm", "timm", "必需"),
    ("opencv-python", "cv2", "必需"),
    ("scikit-learn", "sklearn", "必需"),
    ("matplotlib", "matplotlib", "必需"),
    ("pandas", "pandas", "必需"),
    ("tqdm", "tqdm", "必需"),
    ("gradio", "gradio", "必需(Demo)"),
    ("pytorch-grad-cam", "pytorch_grad_cam", "必需(可视化)"),
    ("Pillow", "PIL", "必需(中文路径读写)"),
]
deps = []
try:
    from importlib.metadata import PackageNotFoundError, packages_distributions, version as dist_version

    module_to_dists = packages_distributions()
except Exception:  # noqa: BLE001
    module_to_dists = {}
    dist_version = None

for pip_name, mod_name, need in DEPS:
    item = {"package": pip_name, "module": mod_name, "need": need}
    try:
        mod = importlib.import_module(mod_name)
        item["status"] = "OK"
        # 优先用「真实已安装的分发包名+版本」，避免把 headless 版误报成 opencv-python
        dists = module_to_dists.get(mod_name.split(".")[0]) or []
        dist = dists[0] if dists else None
        ver = None
        if dist and dist_version is not None:
            try:
                ver = dist_version(dist)
            except PackageNotFoundError:
                ver = None
        item["dist"] = dist or pip_name
        item["version"] = ver or getattr(mod, "__version__", "?")
        item["path"] = getattr(mod, "__file__", "")
    except Exception as exc:  # noqa: BLE001
        item["status"] = "MISSING"
        item["dist"] = pip_name
        item["version"] = None
        item["error"] = f"{type(exc).__name__}: {exc}"
    deps.append(item)
REPORT["dependencies"] = deps

# ---------- 打印 ----------
py = REPORT["python"]
print("=" * 72)
print("Python")
print("=" * 72)
print(f"  version    : {py['version']}  ({py['arch']})")
print(f"  executable : {py['executable']}")
print(f"  platform   : {py['platform']}")

print()
print("=" * 72)
print("PyTorch / CUDA / GPU")
print("=" * 72)
t = REPORT["torch"]
if "error" in t:
    print(f"  [ERROR] {t['error']}")
else:
    print(f"  torch                  : {t['torch_version']}")
    print(f"  torch.version.cuda     : {t['torch_cuda_version']}")
    print(f"  cuDNN                  : {t['cudnn_version']}")
    print(f"  torch.cuda.is_available: {t['cuda_available']}")
    print(f"  device_count           : {t['device_count']}")
    for d in t["devices"]:
        print(
            f"  GPU{d['index']} : {d['name']} | {d['total_memory_GiB']} GiB | "
            f"cc {d['compute_capability']} | SM {d['multi_processor_count']}"
        )
    if t["cuda_available"]:
        print(f"  GPU matmul smoke test  : {'PASS' if t['matmul_smoke_test'] else 'FAIL ' + t.get('matmul_error', '')}")

print()
print("=" * 72)
print("依赖清单")
print("=" * 72)
missing = []
for d in deps:
    if d["status"] == "OK":
        print(f"  [OK]      {d['dist']:<24s} {d['version']}")
    else:
        missing.append(d["package"])
        print(f"  [MISSING] {d['package']:<18s} ({d['need']})  {d.get('error','')}")
print()
if missing:
    print(f"缺失依赖({len(missing)}): {', '.join(missing)}")
else:
    print("全部依赖已安装。")

with open("outputs/env_check.json", "w", encoding="utf-8-sig") as f:
    json.dump(REPORT, f, ensure_ascii=False, indent=2)
print("\n[saved] outputs/env_check.json")
