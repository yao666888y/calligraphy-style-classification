# -*- coding: utf-8 -*-
"""
Step 1-3 / 1-4 / 1-5：数据体检

功能
----
1. 扫描 data 目录，识别 train / test 划分与类别目录；
2. 统计总图片数、类别数、每类 train/test 样本数，找出最多/最少类别；
3. 逐张读取 JPEG 文件头，统计尺寸、通道数（PIL mode）；
4. 分层抽样，用边缘像素均值判断「白底黑字 / 黑底白字」是否混杂；
5. 解析 data/Summary.csv（实际是 xlsx，且中文被二次编码损坏）还原中文名，
   生成 outputs/label_map.json；
6. 每类随机抽 3 张原图拼成 outputs/dataset_preview.png。

说明：本环境缺少 pandas / numpy 之外的分析包，统一用标准库 + PIL + numpy 实现；
严禁 cv2.imread / cv2.imwrite，全部使用 PIL.Image 读写中文路径。
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

# --------------------------------------------------------------------------------------
# 全局配置
# --------------------------------------------------------------------------------------
DATA_ROOT = Path(r"G:\深度学习期末报告\data")
OUT_DIR = Path(r"G:\深度学习期末报告\outputs")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BG_SAMPLE_PER_CLASS = 40          # 每类用于背景检测的抽样数
DUP_SAMPLE_PER_SPLIT = 3000       # 每个划分用于重复检查的抽样数
HEADER_SCAN_ALL = True            # 是否逐张扫描全部图片头（False 则每类抽样 200 张）
PREVIEW_N = 3                     # 每类抽样张数
PREVIEW_CLASSES_PER_ROW = 2       # 网格每行放几个类别
SEED = 42

# Summary.csv 的中文名被「UTF-8 字节按 GBK 解码」二次编码损坏，且每个名字的
# 最后一个汉字丢失了第 3 个字节（日志里显示为「王羲乀」「颜真區」这类错误字）。
# 下面 15 个名字是按「已还原的字节前缀 + 书法家常识」逐个人工确认的，其余 5 个
# （mf 米芾 / hy 弘一 / bdsr 八大山人 / gj 管峻 / lx 鲁迅）可完全自动还原。
NAME_VERIFIED = {
    "wxz": "王羲之", "yzq": "颜真卿", "lgq": "柳公权", "sgt": "孙过庭",
    "smh": "沙孟海", "htj": "黄庭坚", "oyx": "欧阳询", "zmf": "赵孟頫",
    "csl": "褚遂良", "wzm": "文征明", "yyr": "于右任", "shz": "宋徽宗",
    "mzd": "毛泽东", "lqs": "梁秋生", "fwq": "范文强",
}

NULL = "-"
warnings: list[str] = []


def warn(msg: str) -> None:
    warnings.append(msg)
    print(f"[WARN] {msg}")


# --------------------------------------------------------------------------------------
# 中文图例：Matplotlib 字体
# --------------------------------------------------------------------------------------
def setup_chinese_font() -> str:
    candidates = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "SimSun", "KaiTi"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen is None:
        warn(f"未找到中文字体 {candidates}，中文标题可能显示为方块")
        chosen = "DejaVu Sans"
    plt.rcParams["font.sans-serif"] = [chosen] + candidates
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    return chosen


# --------------------------------------------------------------------------------------
# 目录扫描
# --------------------------------------------------------------------------------------
def is_split_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    return len(subdirs) >= 2


def find_splits(root: Path) -> dict[str, Path]:
    """识别 train / test 目录。约定：目录名含 train/test(或 val) 且其子目录即类别。"""
    splits: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if not is_split_dir(p):
            continue
        name = p.name.lower()
        key = None
        if "train" in name:
            key = "train"
        elif "test" in name or "val" in name:
            key = "test"
        if key and key not in splits:
            splits[key] = p
    return splits


def list_images(d: Path) -> list[Path]:
    return sorted(
        (p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS),
        key=lambda p: str(p).lower(),
    )


# --------------------------------------------------------------------------------------
# Summary.csv（xlsx，中文二次编码损坏）解析
# --------------------------------------------------------------------------------------
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col_index(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref).group(1)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _repair_mojibake(s: str) -> tuple[str, bool]:
    """把「UTF-8 字节被按 GBK 解码」的乱码还原。返回 (文本, 是否完整还原)。

    部分字符串末尾有一个 ASCII '?' 吞掉了某个汉字的第 3 个字节，且后面粘着 Total 数字。
    """
    m = re.match(r"^(.*?)(\d{2,})$", s)
    body = m.group(1) if m else s
    for enc in ("gb18030", "gbk", "cp936"):
        try:
            return body.encode(enc).decode("utf-8"), True
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    # 丢字节：把 '?' 换成候选字节再解
    if "?" in body:
        idx = body.index("?")
        prefix = body[:idx]
        suffix = body[idx + 1:]
        try:
            raw = prefix.encode("gb18030")
        except UnicodeEncodeError:
            return body, False
        try:
            tail = suffix.encode("gb18030")
        except UnicodeEncodeError:
            tail = b""
        for b in range(0x80, 0xC0):
            try:
                return (raw + bytes([b]) + tail).decode("utf-8"), False
            except UnicodeDecodeError:
                continue
    return body, False


def find_summary(root: Path) -> Path | None:
    """查找 Summary.csv / Summary.xlsx（Excel 打开时会留下 ~$Summary.xlsx 临时文件）。"""
    if not root.is_dir():
        return None
    for p in sorted(root.iterdir()):
        if p.is_file() and p.stem.lower() == "summary" and p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            return p
    return None


def _rows_from_xlsx(raw: bytes) -> list[list[str]]:
    """解析 xlsx 字节流为二维字符串表。"""
    import io

    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = [
                "".join(t.text or "" for t in si.iter(f"{XLSX_NS}t"))
                for si in root.findall(f"{XLSX_NS}si")
            ]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row in sheet.iter(f"{XLSX_NS}row"):
        cells: dict[int, str] = {}
        for c in row.findall(f"{XLSX_NS}c"):
            t = c.get("t")
            v = c.find(f"{XLSX_NS}v")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif v is not None:
                val = v.text
            else:
                val = ""
            cells[_col_index(c.get("r"))] = val
        if cells:
            width = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(width)])
    return rows


def parse_summary(path: Path) -> dict[str, dict]:
    """返回 {code: {name, total, train, test}}，失败返回 {}。

    兼容两种真实情况：真正的 CSV（utf-8-sig / gb18030），以及「伪装成 CSV 的 xlsx」。
    """
    if path is None or not path.exists():
        warn("未找到 Summary.csv / Summary.xlsx，中文名将标记为「待补充」")
        return {}
    raw = path.read_bytes()
    try:
        if raw[:2] == b"PK":           # xlsx = zip
            rows = _rows_from_xlsx(raw)
        else:
            text = None
            for enc in ("utf-8-sig", "gb18030", "utf-16"):
                try:
                    text = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if text is None:
                warn(f"{path.name} 编码无法识别，跳过中文名解析")
                return {}
            rows = [ln.split(",") for ln in text.splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001
        warn(f"{path.name} 解析失败（{type(exc).__name__}: {exc}），跳过中文名解析")
        return {}

    header = [h.strip() for h in rows[0]] if rows else []
    if not header or "label" not in header[0].lower():
        warn("Summary.csv 表头不符合预期，跳过")
        return {}

    out: dict[str, dict] = {}
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        code = r[0].strip().lower()
        raw_name = r[1] if len(r) > 1 else ""
        name, complete = _repair_mojibake(raw_name)
        nums = [x for x in r[2:] if str(x).strip().isdigit()]
        glued = re.search(r"(\d{2,})$", raw_name)
        if len(nums) >= 3:
            total, train, test = nums[0], nums[1], nums[2]
        elif len(nums) == 2:
            total = glued.group(1) if glued else None
            train, test = nums[0], nums[1]
        else:
            total = glued.group(1) if glued else None
            train, test = (nums + [None, None])[:2]

        if not complete:
            fixed = NAME_VERIFIED.get(code)
            if fixed:
                print(f"[NAME] {code}: 字节残缺（自动得到 {name!r}）→ 已确认中文名 {fixed!r}")
                name = fixed
            else:
                warn(f"{code} 中文名未能完整还原：{name!r}（已标为待补充）")
                name = "待补充"
        out[code] = {
            "name": name,
            "name_source": "summary" if complete else ("summary+verified" if code in NAME_VERIFIED else "missing"),
            "total": int(total) if total else None,
            "train": int(train) if train else None,
            "test": int(test) if test else None,
            "repaired_complete": complete,
        }
    return out


# --------------------------------------------------------------------------------------
# 类别目录名互转
# --------------------------------------------------------------------------------------
SPLIT_RE = re.compile(r"[-_\s.]+")


def class_keys(name: str) -> set[str]:
    s = name.strip().lower()
    parts = [p for p in SPLIT_RE.split(s) if p]
    keys = {s, "".join(parts)}
    if len(parts) >= 2:
        keys.add(parts[0])
        keys.add(parts[0][0] + parts[1])
        keys.add("".join(p[0] for p in parts))
    return keys


def match_classes(train_names: list[str], test_names: list[str]) -> dict[str, str]:
    """test 目录名 -> train 目录名"""
    mapping: dict[str, str] = {}
    unused = list(train_names)
    for t in test_names:
        tk = class_keys(t)
        exact = [c for c in unused if c.lower() == t.lower()]
        if exact:
            mapping[t] = exact[0]
            unused.remove(exact[0])
            continue
        scored = sorted(
            ((-len(tk & class_keys(c)), c) for c in unused if tk & class_keys(c)),
        )
        if scored:
            mapping[t] = scored[0][1]
            unused.remove(scored[0][1])
        else:
            warn(f"测试类别 {t!r} 找不到对应的训练类别")
    return mapping


# --------------------------------------------------------------------------------------
# 图片统计
# --------------------------------------------------------------------------------------
def scan_headers(files: list[Path]) -> dict:
    sizes: Counter = Counter()
    modes: Counter = Counter()
    bad: list[str] = []
    t0 = time.perf_counter()
    for i, p in enumerate(files):
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{p.name}: {type(exc).__name__}")
        if (i + 1) % 20000 == 0:
            print(f"    ...headers {i + 1}/{len(files)}  ({time.perf_counter() - t0:.1f}s)")
    return {
        "sizes": sizes,
        "modes": modes,
        "bad": bad,
        "seconds": round(time.perf_counter() - t0, 2),
    }


def border_features(path: Path) -> tuple[float, float, float, float]:
    """返回 (边缘像素均值, 边缘标准差, 全图均值, 暗像素占比)。

    边缘均值判断底色（白纸/黑纸），暗像素占比作为交叉验证：
    白底单字图像的暗像素一般 5%~25%；若边缘全白但中心大面积发黑，多半是拓片或反色图。
    """
    with Image.open(path) as im:
        g = im.convert("L")
        g.thumbnail((160, 160), Image.BILINEAR)
        a = np.asarray(g, dtype=np.float32)
    h, w = a.shape
    b = max(2, int(round(min(h, w) * 0.06)))
    mask = np.zeros_like(a, dtype=bool)
    mask[:b, :] = True
    mask[-b:, :] = True
    mask[:, :b] = True
    mask[:, -b:] = True
    edge = a[mask]
    dark_ratio = float((a < 80).sum() / a.size)
    return float(edge.mean()), float(edge.std()), float(a.mean()), dark_ratio


def judge_bg(edge_mean: float) -> str:
    if edge_mean >= 170.0:
        return "白底"
    if edge_mean <= 85.0:
        return "黑底"
    return "中间灰"


def sample_bg(files: list[Path], n: int, rng: random.Random) -> dict:
    pick = files if len(files) <= n else rng.sample(files, n)
    rows = []
    for p in pick:
        try:
            em, es, gm, dr = border_features(p)
        except Exception as exc:  # noqa: BLE001
            warn(f"背景检测失败 {p.name}: {type(exc).__name__}: {exc}")
            continue
        bg = judge_bg(em)
        suspect = (bg == "白底" and dr >= 0.45) or (bg == "黑底" and dr <= 0.20)
        rows.append(
            {
                "file": p.name,
                "edge_mean": round(em, 2),
                "edge_std": round(es, 2),
                "global_mean": round(gm, 2),
                "dark_ratio": round(dr, 3),
                "bg": bg,
                "suspect": bool(suspect),
            }
        )
    counts = Counter(r["bg"] for r in rows)
    return {
        "n": len(rows),
        "counts": dict(counts),
        "edge_mean_avg": round(float(np.mean([r["edge_mean"] for r in rows])), 2) if rows else None,
        "global_mean_avg": round(float(np.mean([r["global_mean"] for r in rows])), 2) if rows else None,
        "dark_ratio_avg": round(float(np.mean([r["dark_ratio"] for r in rows])), 3) if rows else None,
        "suspect_count": int(sum(r["suspect"] for r in rows)),
        "suspect_files": [r["file"] for r in rows if r["suspect"]][:10],
        "samples": rows,
    }


def duplicate_check(train_files: list[Path], test_files: list[Path], n_sample: int) -> dict:
    """抽样做精确重复检查（同一张图是否同时出现在 train 与 test，造成数据泄漏）。"""
    import hashlib

    rng = random.Random(SEED)

    def digest(p: Path) -> str:
        return hashlib.md5(p.read_bytes()).hexdigest()

    sub_tr = rng.sample(train_files, min(n_sample, len(train_files)))
    sub_te = rng.sample(test_files, min(n_sample, len(test_files)))
    h_tr: dict[str, list[str]] = defaultdict(list)
    h_te: dict[str, list[str]] = defaultdict(list)
    for p in sub_tr:
        h_tr[digest(p)].append(str(p))
    for p in sub_te:
        h_te[digest(p)].append(str(p))
    common = set(h_tr) & set(h_te)
    return {
        "n_train_sampled": len(sub_tr),
        "n_test_sampled": len(sub_te),
        "cross_split_duplicates": len(common),
        "within_train_duplicates": sum(1 for v in h_tr.values() if len(v) > 1),
        "examples": [{"train": h_tr[h][0], "test": h_te[h][0]} for h in list(common)[:5]],
    }


# --------------------------------------------------------------------------------------
# 预览网格
# --------------------------------------------------------------------------------------
def build_preview(entries: list[dict], out_path: Path) -> None:
    """每类抽 3 张，按「一行 2 个类别 × 每类 3 张」拼成网格。

    类名标题用 fig.text 放在该块左边缘上方（左对齐），避免居中标题被画布裁掉。
    """
    n_cls = len(entries)
    per_row = PREVIEW_CLASSES_PER_ROW
    cols = PREVIEW_N * per_row
    rows = int(np.ceil(n_cls / per_row))
    cell = 2.3
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * cell, rows * (cell + 0.52) + 1.0), dpi=150
    )
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for i, e in enumerate(entries):
        r, c0 = divmod(i, per_row)
        c0 *= PREVIEW_N
        for j, p in enumerate(e["preview"]):
            ax = axes[r, c0 + j]
            try:
                with Image.open(p) as im:
                    ax.imshow(im.convert("RGB"), interpolation="lanczos")
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"读取失败\n{type(exc).__name__}", ha="center", va="center")

    fig.suptitle(
        "中国历代书法名家数据集 · 抽样预览（每类随机 3 张，原图 64×64，白底黑字）",
        fontsize=20,
        y=0.997,
    )
    fig.tight_layout(rect=(0.004, 0.012, 1, 0.976))

    # 网格定位后，再按块写类名（必须在 tight_layout 之后取 bbox）
    for i, e in enumerate(entries):
        r, c0 = divmod(i, per_row)
        c0 *= PREVIEW_N
        bbox = axes[r, c0].get_position()
        fig.text(
            bbox.x0,
            bbox.y1 + 0.004,
            f"{i:>2}. {e['name']}（{e['code']}）    训练 {e['n_train']} / 测试 {e['n_test']}",
            ha="left",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1a1a1a",
        )

    fig.text(
        0.004,
        0.003,
        "数据来源 data/ ；每张图为单个汉字的 64×64 JPEG，放大显示；"
        "中文名由 data/Summary.csv 还原（该文件中文存在二次编码损坏，已逐项校验）",
        fontsize=8,
        color="#555555",
    )
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------
def main() -> int:
    t_start = time.perf_counter()
    font_name = setup_chinese_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[font] 中文字体: {font_name}")

    if not DATA_ROOT.exists():
        print(f"[ERROR] 数据目录不存在: {DATA_ROOT}")
        return 1

    # ---------- 1. 目录结构 ----------
    splits = find_splits(DATA_ROOT)
    print("\n" + "=" * 78)
    print("1. 目录层级结构")
    print("=" * 78)
    print(f"数据根目录: {DATA_ROOT}")
    for key, p in splits.items():
        print(f"  [{key}] {p.relative_to(DATA_ROOT)}  (含 {len([d for d in p.iterdir() if d.is_dir()])} 个类别子目录)")
    if "train" not in splits:
        warn("未识别到 train 目录")
    if "test" not in splits:
        warn("未识别到 test 目录")
    if not splits:
        print("[ERROR] 未识别到任何 train/test 划分")
        return 1

    train_dir = splits.get("train")
    test_dir = splits.get("test")
    train_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir()) if train_dir else []
    test_names = sorted(d.name for d in test_dir.iterdir() if d.is_dir()) if test_dir else []
    print(f"  train 类别数: {len(train_names)}   test 类别数: {len(test_names)}")
    print(f"  train 类别名: {train_names}")
    print(f"  test  类别名: {test_names}")
    if set(train_names) != set(test_names):
        warn("train 与 test 的类别文件夹命名不一致，已按拼音缩写规则自动对齐（见 label_map 的 dir_test）")

    tmap = match_classes(train_names, test_names)  # test 名 -> train 名

    # ---------- 文件清单 ----------
    print("\n[scan] 枚举图片文件 ...")
    per_class: dict[str, dict] = {}
    for tr_name in train_names:
        per_class[tr_name] = {"name": tr_name, "train": list_images(train_dir / tr_name), "test": []}
    for te_name in test_names:
        tr_name = tmap.get(te_name)
        files = list_images(test_dir / te_name)
        if tr_name and tr_name in per_class:
            per_class[tr_name]["test"] = files
            per_class[tr_name]["test_dir"] = te_name
        else:
            per_class[te_name] = {"name": te_name, "train": [], "test": files, "test_dir": te_name}
    for tr_name in train_names:
        per_class[tr_name].setdefault("test_dir", tr_name)

    total_files = sum(len(v["train"]) + len(v["test"]) for v in per_class.values())
    print(f"[scan] 图片总数: {total_files}")

    # ---------- Summary 中文名 ----------
    summary_path = find_summary(DATA_ROOT)
    summary = parse_summary(summary_path)
    if summary_path:
        print(f"[summary] 使用 {summary_path.name}（{len(summary)} 条类别记录）")

    # ---------- 2. 每类数量 ----------
    print("\n" + "=" * 78)
    print("2. 类别与样本数")
    print("=" * 78)
    table = []
    for code, v in per_class.items():
        n_tr, n_te = len(v["train"]), len(v["test"])
        s = summary.get(code.lower(), {})
        table.append(
            {
                "code": code,
                "name": s.get("name") or "待补充",
                "n_train": n_tr,
                "n_test": n_te,
                "n_total": n_tr + n_te,
                "summary_train": s.get("train"),
                "summary_test": s.get("test"),
                "summary_total": s.get("total"),
                "test_dir": v["test_dir"],
            }
        )
    table.sort(key=lambda x: x["code"].lower())
    n_cls = len(table)
    tot_train = sum(t["n_train"] for t in table)
    tot_test = sum(t["n_test"] for t in table)

    print(f"{'#':>2} {'code':<6} {'中文名':<7} {'train':>6} {'test':>5} {'total':>6}  {'Summary(train/test)':>22}")
    for i, t in enumerate(table):
        ref = f"{t['summary_train']}/{t['summary_test']}" if t["summary_train"] else NULL
        print(f"{i:>2} {t['code']:<6} {t['name']:<7} {t['n_train']:>6} {t['n_test']:>5} {t['n_total']:>6}  {ref:>22}")

    print(f"\n  类别数: {n_cls}")
    print(f"  训练集: {tot_train}   测试集: {tot_test}   合计: {tot_train + tot_test}")
    mx = max(table, key=lambda x: x["n_total"])
    mn = min(table, key=lambda x: x["n_total"])
    print(f"  样本最多: {mx['name']}({mx['code']}) {mx['n_total']}")
    print(f"  样本最少: {mn['name']}({mn['code']}) {mn['n_total']}")
    print(f"  最多/最少 = {mx['n_total'] / max(1, mn['n_total']):.2f} 倍（类别不均衡程度）")

    # 与 Summary 交叉校验
    mismatches = [
        t for t in table
        if t["summary_train"] is not None
        and (t["n_train"] != t["summary_train"] or t["n_test"] != t["summary_test"])
    ]
    if mismatches:
        for t in mismatches:
            warn(
                f"{t['code']} 实际 {t['n_train']}/{t['n_test']} 与 Summary 记录 "
                f"{t['summary_train']}/{t['summary_test']} 不一致"
            )
    else:
        src = summary_path.name if summary_path else "Summary"
        print(f"  [OK] 每类 train/test 数量与 {src} 记录完全一致")

    # ---------- 2.5 重复 / 泄漏检查 ----------
    print("\n[check] 跨 train/test 精确重复抽样检查 ...")
    dup = duplicate_check(
        [p for v in per_class.values() for p in v["train"]],
        [p for v in per_class.values() for p in v["test"]],
        DUP_SAMPLE_PER_SPLIT,
    )
    print(
        f"  抽样 train {dup['n_train_sampled']} / test {dup['n_test_sampled']}："
        f"跨集完全相同 {dup['cross_split_duplicates']} 张，train 内重复 {dup['within_train_duplicates']} 组"
    )
    if dup["cross_split_duplicates"]:
        warn("train 与 test 存在完全相同的图片，可能造成数据泄漏")
    else:
        print("  [OK] 抽样未发现 train/test 重复（未见明显数据泄漏）")

    # ---------- 3. 尺寸 / 通道 ----------
    print("\n" + "=" * 78)
    print("3. 图片尺寸与通道数")
    print("=" * 78)
    all_files = [p for v in per_class.values() for p in v["train"]] + [
        p for v in per_class.values() for p in v["test"]
    ]
    if not HEADER_SCAN_ALL:
        rng0 = random.Random(SEED)
        all_files = (
            [p for v in per_class.values() for p in rng0.sample(v["train"], min(200, len(v["train"])))]
            if len(all_files) > 5000
            else all_files
        )
    print(f"[scan] 读取 {len(all_files)} 张图片的文件头 ...")
    hdr = scan_headers(all_files)
    sizes: Counter = hdr["sizes"]
    modes: Counter = hdr["modes"]
    all_w = [w for (w, h), c in sizes.items() for _ in range(c)]
    all_h = [h for (w, h), c in sizes.items() for _ in range(c)]
    print(f"  尺寸种类: {len(sizes)} 种")
    for (w, h), c in sizes.most_common(10):
        print(f"    {w}x{h}: {c} 张 ({c / len(all_files) * 100:.2f}%)")
    print(f"  宽: min={min(all_w)}  max={max(all_w)}  中位={int(np.median(all_w))}")
    print(f"  高: min={min(all_h)}  max={max(all_h)}  中位={int(np.median(all_h))}")
    mode_channels = {"1": 1, "L": 1, "P": 1, "LA": 2, "RGB": 3, "RGBA": 4, "CMYK": 4, "YCbCr": 3, "I": 1, "F": 1}
    print("  通道(PIL mode):")
    for m, c in modes.most_common():
        print(f"    {m:<6} {mode_channels.get(m, '?')} 通道  {c} 张 ({c / len(all_files) * 100:.2f}%)")
    if hdr["bad"]:
        warn(f"有 {len(hdr['bad'])} 张图片无法打开: {hdr['bad'][:5]}")
    uniform_size = len(sizes) == 1

    # ---------- 4. 黑底/白底 ----------
    print("\n" + "=" * 78)
    print("4. 底色抽样检测（边缘像素均值）")
    print("=" * 78)
    rng = random.Random(SEED)
    bg_all: dict[str, dict] = {}
    overall = Counter()
    for code, v in per_class.items():
        files = v["train"] if v["train"] else v["test"]
        bg = sample_bg(files, BG_SAMPLE_PER_CLASS, rng)
        bg_all[code] = bg
        overall.update(bg["counts"])
    total_bg = sum(overall.values())
    print(f"  抽样 {total_bg} 张；判定规则: 边缘 6% 像素灰度均值 >=170 白底, <=85 黑底, 其余中间灰")
    for k in ("白底", "黑底", "中间灰"):
        c = overall.get(k, 0)
        print(f"    {k}: {c} 张 ({c / max(1, total_bg) * 100:.1f}%)")
    all_dark = [s["dark_ratio"] for bg in bg_all.values() for s in bg["samples"]]
    all_gmean = [s["global_mean"] for bg in bg_all.values() for s in bg["samples"]]
    n_suspect = sum(bg["suspect_count"] for bg in bg_all.values())
    print(f"  暗像素占比: 均值 {np.mean(all_dark):.3f}  范围 [{min(all_dark):.3f}, {max(all_dark):.3f}]")
    print(f"  全图灰度均值: 均值 {np.mean(all_gmean):.1f}  范围 [{min(all_gmean):.1f}, {max(all_gmean):.1f}]")
    print(f"  极性存疑样本(边缘白但中心大面积发黑 / 边缘黑但几乎无亮部): {n_suspect} 张")

    print("\n  分类别（白/黑/灰, 边缘均值, 暗像素占比）:")
    mixed_classes = []
    for code, bg in sorted(bg_all.items()):
        cc = bg["counts"]
        w_, b_, g_ = cc.get("白底", 0), cc.get("黑底", 0), cc.get("中间灰", 0)
        flag = ""
        if w_ >= 3 and b_ >= 3:
            mixed_classes.append(code)
            flag = "  <== 黑白混杂"
        print(
            f"    {code:<6} 白{w_:>3} 黑{b_:>3} 灰{g_:>3}  "
            f"边缘均值={bg['edge_mean_avg']:>6.1f}  暗占比={bg['dark_ratio_avg']:.3f}{flag}"
        )
    if mixed_classes:
        warn(f"以下类别内部同时存在黑底与白底样本: {mixed_classes}")
    else:
        print("\n  [OK] 各类别内部底色一致（未发现类内黑白混杂）")
    if overall.get("黑底", 0) and overall.get("白底", 0):
        print("  注意: 整体数据集中黑底与白底并存，训练时建议统一反色或做极性归一化")

    # ---------- 5. label_map.json ----------
    label_map = {}
    for i, t in enumerate(table):
        label_map[str(i)] = {
            "code": t["code"],
            "name": t["name"],
            "train_dir": t["code"],
            "test_dir": t["test_dir"],
        }
    lm_path = OUT_DIR / "label_map.json"
    lm_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"\n[out] {lm_path}  ({len(label_map)} 类)")

    # ---------- 6. 预览网格 ----------
    print("[out] 生成预览网格 ...")
    entries = []
    prng = random.Random(SEED)
    for i, t in enumerate(table):
        v = per_class[t["code"]]
        pool = v["train"] if v["train"] else v["test"]
        picks = prng.sample(pool, min(PREVIEW_N, len(pool))) if pool else []
        entries.append(
            {
                "code": t["code"],
                "name": t["name"],
                "n_train": t["n_train"],
                "n_test": t["n_test"],
                "preview": picks,
            }
        )
    pv_path = OUT_DIR / "dataset_preview.png"
    build_preview(entries, pv_path)
    with Image.open(pv_path) as im:
        pv_size = im.size
    print(f"[out] {pv_path}  ({pv_size[0]}x{pv_size[1]})")

    # ---------- 7. 报告 JSON ----------
    report = {
        "data_root": str(DATA_ROOT),
        "structure": {
            "train_dir": str(train_dir.relative_to(DATA_ROOT)) if train_dir else None,
            "test_dir": str(test_dir.relative_to(DATA_ROOT)) if test_dir else None,
            "n_classes": n_cls,
            "class_folders": {"train": train_names, "test": test_names},
            "folder_align": tmap,
        },
        "counts": {
            "total": tot_train + tot_test,
            "train": tot_train,
            "test": tot_test,
            "per_class": table,
            "max_class": {"code": mx["code"], "total": mx["n_total"]},
            "min_class": {"code": mn["code"], "total": mn["n_total"]},
            "imbalance_ratio": round(mx["n_total"] / max(1, mn["n_total"]), 3),
        },
        "duplicates": dup,
        "images": {
            "n_scanned": len(all_files),
            "n_size_kinds": len(sizes),
            "sizes_top": [{"size": f"{w}x{h}", "count": c} for (w, h), c in sizes.most_common(20)],
            "uniform_size": uniform_size,
            "width": {"min": min(all_w), "max": max(all_w), "median": int(np.median(all_w))},
            "height": {"min": min(all_h), "max": max(all_h), "median": int(np.median(all_h))},
            "modes": [{"mode": m, "channels": mode_channels.get(m), "count": c} for m, c in modes.most_common()],
            "bad_files": hdr["bad"][:20],
            "scan_seconds": hdr["seconds"],
        },
        "background": {
            "rule": "边缘 6% 像素的灰度均值：>=170 白底，<=85 黑底，其余中间灰；暗像素=灰度<80",
            "sample_per_class": BG_SAMPLE_PER_CLASS,
            "overall": dict(overall),
            "n_sampled": total_bg,
            "dark_ratio": {
                "avg": round(float(np.mean(all_dark)), 4),
                "min": round(float(min(all_dark)), 4),
                "max": round(float(max(all_dark)), 4),
            },
            "global_mean": {
                "avg": round(float(np.mean(all_gmean)), 2),
                "min": round(float(min(all_gmean)), 2),
                "max": round(float(max(all_gmean)), 2),
            },
            "polarity_suspect": n_suspect,
            "per_class": {
                k: {
                    "counts": v["counts"],
                    "edge_mean_avg": v["edge_mean_avg"],
                    "dark_ratio_avg": v["dark_ratio_avg"],
                    "suspect_count": v["suspect_count"],
                    "suspect_files": v["suspect_files"],
                }
                for k, v in bg_all.items()
            },
            "mixed_within_class": mixed_classes,
        },
        "summary_csv": {
            "file": summary_path.name if summary_path else None,
            "parsed": bool(summary),
            "records": summary,
            "counts_match": not mismatches,
        },
        "label_map_file": str(lm_path),
        "preview_file": str(pv_path),
        "warnings": warnings,
        "elapsed_seconds": round(time.perf_counter() - t_start, 1),
    }
    rp_path = OUT_DIR / "dataset_report.json"
    rp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    print(f"[out] {rp_path}")

    print(f"\n[done] 用时 {report['elapsed_seconds']}s")
    if warnings:
        print(f"[warn] 共 {len(warnings)} 条警告，详见 dataset_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
