# -*- coding: utf-8 -*-
"""书法风格诊断 · 本地演示服务。

接口
----
GET  /          返回 index.html
GET  /models    返回 [{"id": "...", "label": "..."}]（只列已存在权重的模型）
POST /predict   表单 file + model → {"top3":[{"name","prob"}...], "cleaned_image":"<base64 png>"}

约定
----
* 模型只在启动时加载一次；每个模型各自的预处理参数（字幅 fill / 是否二值化）
  由 MODEL_SPECS 指定，保证「推理时的样子」与「它自己训练时的样子」一致
* 中文名一律从 outputs/label_map.json 读
* 读图用 PIL（中文路径安全），绝不使用 cv2.imread / cv2.imwrite
* 图像归一化统一走 utils/preprocess.py 的 normalize_calligraphy_image（训练/推理共用）
"""
from __future__ import annotations

import base64
import csv
import io
import json
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.networks import build_model  # noqa: E402
from utils.dataset import NORM_STATS  # noqa: E402
from utils.preprocess import fill_small_holes, legacy_dataset_style, normalize_calligraphy_image  # noqa: E402

LABEL_MAP = ROOT / "outputs" / "label_map.json"
WEB_DIR = ROOT / "outputs" / "web_test"
RESULTS_CSV = WEB_DIR / "results.csv"

IMG_SIZE = 128
NORM = "imagenet"
CLEAN_MAX_SIDE = 2000          # 手机原图先缩到这个尺寸再处理，避免大图太慢

# 每个模型自己的归一化参数：必须与它训练时看到的分布一致
#   fill     : 汉字长边占画布比例
#   binarize : True → 纯黑纯白（域对齐组）；False → 保留笔画软边灰度（第 1 组）
MODEL_SPECS = [
    {
        # 第 6 组：实测最强（官方对齐 98.59%）。预处理用「填石花 + 64 往返」，
        # 实测能把碑刻风格「雪」从沙孟海 85% 纠正为颜真卿 84%
        "id": "exp6_convnext_224",
        "label": "第6组·域对齐 ConvNeXt-224",
        "ckpt": ROOT / "checkpoints" / "exp6_convnext_224" / "best.pth",
        "arch": "convnext_tiny",
        "mode": "domain_hole",
        "size": 224,
        "fill": 0.88,
        "binarize": True,
        "mask_background": True,
    },
    {
        "id": "exp1_resnet18_base",
        "label": "第 1 组 · 普通学法",
        "ckpt": ROOT / "checkpoints" / "exp1_resnet18_base" / "best.pth",
        "arch": "resnet18",
        "mode": "web_v1",           # 紧致裁切 + 照片 0.88 / 小图 0.64
        "size": 128,
        "fill": 0.64,
        "binarize": False,
        "mask_background": "auto",
    },
]

CSV_HEADER = ["时间", "原图文件名", "模型", "第一名", "第一名概率", "前三名及概率"]

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

_state: dict = {"models": {}, "classes": [], "device": torch.device("cpu")}
_seq_lock = threading.Lock()
_seq = 0


# --------------------------------------------------------------------------------------
# 模型加载（只执行一次）
# --------------------------------------------------------------------------------------
def load_models_once() -> None:
    label_map = json.loads(LABEL_MAP.read_text(encoding="utf-8-sig"))
    names = [label_map[str(i)]["name"] for i in range(len(label_map))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = NORM_STATS[NORM]
    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    loaded = {}
    for spec in MODEL_SPECS:
        if not spec["ckpt"].exists():
            print(f"[model] 跳过 {spec['id']}（没有权重）")
            continue
        ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
        model = build_model(spec["arch"], num_classes=len(names), pretrained=False)
        model.load_state_dict(ck["model"])
        model.eval().to(device)
        loaded[spec["id"]] = {
            "model": model,
            "label": spec["label"],
            "mode": spec.get("mode", "domain"),
            "size": spec.get("size", IMG_SIZE),
            "fill": spec["fill"],
            "binarize": spec["binarize"],
            "mask_background": spec.get("mask_background", True),
            "epoch": ck.get("epoch"),
            "val_f1": ck.get("val_macro_f1"),
        }
        print(f"[model] 已加载 {spec['id']} (epoch={ck.get('epoch')}, "
              f"val_macro_f1={ck.get('val_macro_f1')}, fill={spec['fill']}, binarize={spec['binarize']})")

    if not loaded:
        raise RuntimeError("没有任何可用权重")

    _state.update(models=loaded, classes=names, device=device, preprocess=preprocess)
    print(f"[model] 类别 {len(names)} 个，设备 {device}，可用模型 {list(loaded)}")


# --------------------------------------------------------------------------------------
# 预处理（薄封装：真正的逻辑在 utils/preprocess.py，训练/推理共用）
# --------------------------------------------------------------------------------------
def clean_photo(
    pil_img: Image.Image,
    fill: float = 0.88,
    binarize: bool = True,
    mask_background: bool = True,
    mode: str = "domain",
    size: int = IMG_SIZE,
) -> tuple[Image.Image, dict]:
    """按模型的预处理方式归一化。

    mode='domain' → 共享归一化（紧致裁切 + fill 字幅 + 可选二值化）
    mode='legacy' → 第 1 组训练时的老取景（原图 + 12% 补白，不裁切）
    """
    rgb = pil_img.convert("RGB")
    if max(rgb.size) > CLEAN_MAX_SIDE:
        s = CLEAN_MAX_SIDE / max(rgb.size)
        rgb = rgb.resize((max(1, int(rgb.width * s)), max(1, int(rgb.height * s))), Image.LANCZOS)

    if mode == "legacy":
        out = legacy_dataset_style(rgb, size=IMG_SIZE, margin=0.12)
        return out, {"mode": "legacy", "input_size": rgb.size, "fill": 0.68,
                     "binarize": False, "ink_ratio": None}

    # 照片级输入（>=200px）用 0.88 字幅：与域对齐训练分布一致，肉眼也不会「变小」；
    # 64px 这类数据集小图仍用 0.64，保住 exp1 的准确率
    if mode == "web_v1" and min(rgb.size) >= 200:
        fill = 0.88

    out, info = normalize_calligraphy_image(
        rgb, size=size, fill=fill, binarize=binarize,
        mask_background=mask_background, return_info=True,
    )

    # domain_hole：先填掉碑刻石花/飞白小孔，再走 64 往返（低通，抹掉残余高频噪点）
    filled = 0
    if mode == "domain_hole":
        out, filled = fill_small_holes(out)
        out = out.resize((64, 64), Image.LANCZOS).resize((size, size), Image.LANCZOS)

    delta = abs(info.ink_mask_ratio - info.ink_src_ratio)
    return out, {
        "mode": mode,
        "size": size,
        "holes_filled": filled,
        "input_size": info.input_size,
        "stroke_px": info.stroke_px,
        "window": info.window,
        "ink_ratio": info.ink_ratio,
        "bbox": info.bbox,
        "crop": info.crop_size,
        "fill": info.fill,
        "binarize": info.binarized,
        "dropped": info.n_components_dropped,
        "frame_dropped": info.frame_dropped,
        "dark_bg": info.dark_bg,
        "ink_src_ratio": info.ink_src_ratio,
        "ink_mask_ratio": info.ink_mask_ratio,
        "ink_final_ratio": info.ink_final_ratio,
        "glyph_fill_actual": info.glyph_fill_actual,
        "ratio_delta": round(delta, 4),
        "ratio_warn": bool(delta > 0.03),
    }


# --------------------------------------------------------------------------------------
# 存档
# --------------------------------------------------------------------------------------
def next_seq() -> int:
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


def save_case(original_name: str, raw_bytes: bytes, cleaned: Image.Image) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = WEB_DIR / f"{stamp}_{next_seq():03d}"
    while case_dir.exists():
        case_dir = WEB_DIR / f"{stamp}_{next_seq():03d}"
    case_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(original_name).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
        suffix = ".img"
    (case_dir / f"original{suffix}").write_bytes(raw_bytes)
    cleaned.save(case_dir / "cleaned.png")
    return case_dir


def append_result(row: list) -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


# --------------------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------------------
@app.get("/")
def index():
    return send_file(ROOT / "index.html")


@app.get("/models")
def models():
    return jsonify([{"id": k, "label": v["label"]} for k, v in _state["models"].items()])


@app.post("/predict")
def predict():
    uploaded = request.files.get("file")
    model_id = request.form.get("model") or next(iter(_state["models"]), "")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "没有收到图片（表单字段名应为 file）"}), 400
    if model_id not in _state["models"]:
        return jsonify({"error": f"未知模型 {model_id}"}), 400

    raw = uploaded.read()
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"图片无法识别: {type(exc).__name__}"}), 400

    spec = _state["models"][model_id]
    try:
        cleaned, info = clean_photo(
            pil,
            fill=spec["fill"],
            binarize=spec["binarize"],
            mask_background=spec["mask_background"],
            mode=spec["mode"],
            size=spec["size"],
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    device = _state["device"]
    tensor = _state["preprocess"](cleaned).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        logits = spec["model"](tensor)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu()

    top = torch.topk(probs, k=3)
    names = _state["classes"]
    top3 = [{"name": names[int(i)], "prob": round(float(p), 6)} for p, i in zip(top.values, top.indices)]

    buf = io.BytesIO()
    cleaned.save(buf, format="PNG")
    png_bytes = buf.getvalue()          # 与落盘 cleaned.png 是同一份字节

    case_dir = save_case(uploaded.filename, raw, cleaned)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_result([
        now,
        uploaded.filename,
        model_id,
        top3[0]["name"],
        f"{top3[0]['prob']:.6f}",
        "; ".join(f"{t['name']}:{t['prob']:.4f}" for t in top3),
    ])

    print(
        f"[predict] {now} {uploaded.filename} [{model_id}] -> {top3[0]['name']} {top3[0]['prob']:.4f}  "
        f"| fill={info['fill']} bin={info['binarize']} dark_bg={info.get('dark_bg')} "
        f"ink src={info.get('ink_src_ratio')} mask={info.get('ink_mask_ratio')} "
        f"delta={info.get('ratio_delta')} | {case_dir.name}"
    )
    if info.get("ratio_warn"):
        # 清洗前后墨迹占比偏差 > 3pp：阈值可能在啃笔画边缘，标红提示复查
        print(f"\033[91m[RED][清洗自检] {uploaded.filename}: 墨迹占比 "
              f"{info.get('ink_src_ratio')} → {info.get('ink_mask_ratio')}，"
              f"偏差 {info.get('ratio_delta'):.3f} > 0.03\033[0m")

    return jsonify({
        "top3": top3,
        "cleaned_image": base64.b64encode(png_bytes).decode("ascii"),
        "case_dir": str(case_dir.relative_to(ROOT)),
    })


def main() -> None:
    load_models_once()
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    print("[serve] http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()


