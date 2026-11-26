# -*- coding: utf-8 -*-
"""中国历代书法名家分类 —— 数据管道（Step 2）。

设计要点（对应考核规范）
------------------------
1. **禁止翻转**：书法单字左右/上下翻转后笔画走势与汉字结构全部失真，属于标签破坏。
   `allow_hflip` 默认 False，仅留作后续「反面消融实验」的开关。
2. **底色统一**：用四边边缘像素均值判断底色，黑底白字自动反色为白底黑字；
   `bg_mode="auto"` 会自动判断，`"white"` 强制视为白底，`"off"` 关闭（原始像素）。
   即使当前数据集全是白底，这段逻辑也照常跑（阈值判断 + 跳过反色），不写死。
3. **等比补白**：先按长边等比缩放，短边两侧补纯白(255) 成正方形，再额外留一圈白边
   （默认 12%），然后缩放到目标尺寸。全程不做非等比拉伸，PIL 用 LANCZOS。
   这圈白边是必需的：本数据集单字裁得极紧（抽样 16 张里 10 张墨迹直接贴边），
   没有余量时 ±8° 旋转就会把笔画切掉（实测曾出现 10/16 张笔画越界）。
4. **增强只用三样**：±8° 旋转（expand=True，画布跟着转、绝不裁切）、轻微缩放平移
   （缩放 0.95~1.05、平移 4%，均在白边余量内）、轻量随机擦除。
   验证/测试集只做：底色统一 → 等比补白 → 缩放 → ToTensor → Normalize，与训练同框。
5. **划分**：原始数据只有 train/test，没有独立 val。因此从 train 按 seed=42 做 9:1
   分层抽样切出 val；**test 全程不参与任何调参**。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.preprocess import normalize_calligraphy_image

# --------------------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# 笔画与墨色都靠灰度对比，ImageNet 统计量对后续换预训练骨干更通用
NORM_STATS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}

# 底色判定阈值：边缘灰度均值 <= BLACK_BG_MAX 视为黑底；>= WHITE_BG_MIN 视为白底
BLACK_BG_MAX = 85.0
WHITE_BG_MIN = 170.0

WHITE = 255


# --------------------------------------------------------------------------------------
# 基础图像处理（全部走 PIL，兼容中文路径；不使用 cv2.imread/imwrite）
# --------------------------------------------------------------------------------------
def border_mean(img: Image.Image) -> float:
    """四边 6% 宽度条带的灰度均值，用于判断底色。"""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    b = max(2, int(round(min(h, w) * 0.06)))
    edge = np.concatenate(
        [g[:b, :].ravel(), g[-b:, :].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()]
    )
    return float(edge.mean())


def detect_background(img: Image.Image) -> tuple[str, float]:
    """返回 (底色, 边缘均值)。底色为 'black' / 'white' / 'gray'。"""
    m = border_mean(img)
    if m <= BLACK_BG_MAX:
        return "black", m
    if m >= WHITE_BG_MIN:
        return "white", m
    return "gray", m


def unify_background(img: Image.Image, mode: str = "auto") -> tuple[Image.Image, str]:
    """把黑底白字反色成白底黑字。

    mode: 'auto' 按边缘均值自动判断 | 'white' 一律当白底（不反色）| 'off' 不做处理
    返回 (处理后的图, 判定结果)，判定结果用于统计「实际反色了多少张」。
    """
    img = img.convert("RGB")
    if mode == "off":
        return img, "off"
    if mode == "white":
        return img, "white"
    bg, _ = detect_background(img)
    if bg == "black":
        # 反色：255 - pixel，黑底白字 → 白底黑字
        return Image.fromarray(255 - np.asarray(img)), "inverted"
    return img, bg


def pad_to_square(img: Image.Image, fill: int = WHITE, margin: float = 0.0) -> Image.Image:
    """按长边等比缩放后居中补白成正方形（严禁拉伸变形），可再留一圈白边。

    margin: 额外白边比例（相对正方形边长）。书法单字原图普遍裁得很紧，
          必须留出 margin 给旋转/平移，否则 ±8° 旋转就会把笔画切掉。
    """
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    m = int(round(side * margin)) if margin > 0 else 0
    canvas_side = side + 2 * m
    if img.size == (canvas_side, canvas_side) and m == 0:
        return img
    canvas = Image.new("RGB", (canvas_side, canvas_side), (fill, fill, fill))
    canvas.paste(img, (m + (side - w) // 2, m + (side - h) // 2))
    return canvas


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------
def build_train_transforms(
    img_size: int = 128,
    norm: str = "imagenet",
    allow_hflip: bool = False,
    rotation_deg: float = 8.0,
    translate: float = 0.04,
    scale_range: tuple[float, float] = (0.95, 1.05),
    erase_p: float = 0.25,
    erase_mode: str = "white",
) -> transforms.Compose:
    """训练增强：±8° 旋转 + 轻微缩放平移 + 轻量随机擦除（全部白底填充）。

    关键：旋转用 expand=True，让画布跟着转、不裁掉任何像素；
    缩放/平移幅度压在 1.05 与 4% 以内，配合 Dataset 预留的 12% 白边，
    保证增强后笔画不出界（实测边距仍 > 0）。

    注意：默认**不含**任何翻转。allow_hflip=True 只用于反面消融实验。
    """
    mean, std = NORM_STATS[norm]
    ops: list = []
    if allow_hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops += [
        transforms.RandomRotation(
            degrees=rotation_deg,
            interpolation=InterpolationMode.BICUBIC,
            expand=True,   # 画布扩张，旋转后四角填白，绝不裁切笔画
            fill=WHITE,
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(translate, translate),
            scale=scale_range,
            interpolation=InterpolationMode.BICUBIC,
            fill=WHITE,
        ),
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if erase_p > 0:
        # 归一化空间里的「白」=(1-mean)/std；「均值」等价于 0
        if erase_mode == "white":
            value = [(1.0 - m) / s for m, s in zip(mean, std)]
        elif erase_mode == "mean":
            value = [0.0, 0.0, 0.0]
        else:
            raise ValueError(f"erase_mode 只支持 'white' / 'mean'，收到 {erase_mode!r}")
        ops.append(
            transforms.RandomErasing(
                p=erase_p,
                # 只擦 1%~6% 面积、细长比 0.5~2.5：128px 单字图里约 20~32px 的小块。
                # 再大就会像「被切了一刀」把整笔画抹掉，反而伤语义（实测 10%/ratio3.3 时即如此）。
                scale=(0.01, 0.06),
                ratio=(0.5, 2.5),
                value=value,
            )
        )
    return transforms.Compose(ops)


def build_eval_transforms(img_size: int = 128, norm: str = "imagenet") -> transforms.Compose:
    """验证/测试：只做尺寸归一 + 标准化，不含任何随机性。"""
    mean, std = NORM_STATS[norm]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


# --------------------------------------------------------------------------------------
# 域对齐模式专用的两项「毛笔实战」增强
# --------------------------------------------------------------------------------------
class RandomFillJitter:
    """随机缩放留白：让汉字长边在 [lo, hi] 之间波动，避免模型死记固定字号。"""

    def __init__(self, base_fill: float = 0.88, lo: float = 0.78, hi: float = 0.94,
                 p: float = 0.7, size: int = 128) -> None:
        self.base_fill, self.lo, self.hi, self.p, self.size = base_fill, lo, hi, p, size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        f = random.uniform(self.lo, self.hi)
        side = max(8, int(round(self.size * f / self.base_fill)))
        scaled = img.resize((side, side), Image.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (WHITE, WHITE, WHITE))
        if side <= self.size:
            off = (self.size - side) // 2
            canvas.paste(scaled, (off, off))
        else:
            left = (side - self.size) // 2
            canvas.paste(scaled.crop((left, left, left + self.size, left + self.size)), (0, 0))
        return canvas


class RandomMorph:
    """轻微形态学腐蚀/膨胀：模拟蘸墨浓淡造成的笔画粗细变化。"""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice([2, 3])
        kernel = np.ones((k, k), np.uint8)
        arr = np.asarray(img.convert("L"))
        arr = cv2.erode(arr, kernel) if random.random() < 0.5 else cv2.dilate(arr, kernel)
        return Image.fromarray(arr).convert("RGB")


class InkCoverageFix:
    """温和版覆盖率纠正：只在墨迹占比越界时往区间内拉。

    二值图里 黑=0 / 白=255：
        cv2.dilate → 白扩张 → 黑变细；cv2.erode → 黑扩张 → 黑变粗。
    """

    def __init__(self, lo: float = 0.10, hi: float = 0.28) -> None:
        self.lo, self.hi = lo, hi

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.asarray(img.convert("L"))
        cov = float((arr < 128).mean())
        if self.lo <= cov <= self.hi:
            return img
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(8):
            arr = cv2.dilate(arr, kernel) if cov > self.hi else cv2.erode(arr, kernel)
            cov = float((arr < 128).mean())
            if self.lo <= cov <= self.hi:
                break
        return Image.fromarray(arr).convert("RGB")


class RandomBlurJPEG:
    """轻量高斯模糊 / JPEG 压缩痕迹（模拟手机拍摄与传输）。"""

    def __init__(self, p: float = 0.3) -> None:
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        import io as _io

        from PIL import ImageFilter

        if random.random() > self.p:
            return img
        if random.random() < 0.5:
            return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.0)))
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(35, 75))
        buf.seek(0)
        with Image.open(buf) as im:
            return im.convert("RGB")


class RandomSpecks:
    """极少量随机细碎噪点：模拟没洗净的米字格残渣。"""

    def __init__(self, p: float = 0.3, n_min: int = 3, n_max: int = 10) -> None:
        self.p, self.n_min, self.n_max = p, n_min, n_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        arr = np.array(img.convert("L"))
        h, w = arr.shape
        for _ in range(random.randint(self.n_min, self.n_max)):
            y, x = random.randrange(h), random.randrange(w)
            r = random.choice([1, 2])
            arr[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = 0
        return Image.fromarray(arr).convert("RGB")


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class CalligraphyDataset(Dataset):
    """书法单字数据集。

    samples: [(图片路径, 类别索引), ...]
    返回: (tensor[C,H,W], label:int)

    预处理顺序：读取 → 底色统一 → 等比补白 → 缩放 → (增强) → ToTensor → Normalize
    """

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform,
        bg_mode: str = "auto",
        img_size: int = 128,
        margin: float = 0.12,
        cache_bg: bool = True,
        return_path: bool = False,
        domain_align: bool = False,
        fill: float = 0.88,
        extra_aug=None,
        cache_root: Path | None = None,
        data_root: Path | None = None,
    ) -> None:
        if not samples:
            raise ValueError("samples 为空，检查数据目录与标签映射")
        self.samples = samples
        self.transform = transform
        self.bg_mode = bg_mode
        self.img_size = img_size
        self.margin = margin
        self.cache_bg = cache_bg
        self.return_path = return_path
        self.domain_align = domain_align
        self.fill = fill
        self.extra_aug = extra_aug
        self.cache_root = Path(cache_root) if cache_root else None
        self.data_root = Path(data_root) if data_root else None
        self._bg_cache: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _unify(self, path: Path, img: Image.Image) -> Image.Image:
        """底色统一（带缓存，避免每个 epoch 重复判断同一张图）。"""
        key = str(path)
        if self.bg_mode == "off":
            return img.convert("RGB")
        if self.cache_bg and key in self._bg_cache:
            state = self._bg_cache[key]
            if state == "inverted":
                return Image.fromarray(255 - np.asarray(img.convert("RGB")))
            return img.convert("RGB")
        out, state = unify_background(img, self.bg_mode)
        if self.cache_bg:
            self._bg_cache[key] = state
        return out

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        cached = None
        if self.cache_root is not None and self.data_root is not None:
            try:
                cand = self.cache_root / Path(path).relative_to(self.data_root)
                cand = cand.with_suffix(".png")
                if cand.exists():
                    cached = cand
            except ValueError:
                cached = None

        with Image.open(cached if cached is not None else path) as im:
            if self.domain_align:
                if cached is not None:
                    img = im.convert("RGB")          # 缓存已是 0.88 字幅二值图
                else:
                    img = normalize_calligraphy_image(
                        im.convert("RGB"), size=self.img_size, fill=self.fill, binarize=True
                    )
            else:
                img = self._unify(path, im)
                # 等比缩放 + 居中补白成正方形 + 预留白边
                img = pad_to_square(img, fill=WHITE, margin=self.margin)
        if self.extra_aug is not None:
            img = self.extra_aug(img)
        tensor = self.transform(img)
        if self.return_path:
            return tensor, label, str(path)
        return tensor, label

    # 统计训练时实际反色了多少张，便于汇报
    def bg_stats(self) -> dict:
        from collections import Counter

        return dict(Counter(self._bg_cache.values()))


# --------------------------------------------------------------------------------------
# 数据集枚举 / 划分
# --------------------------------------------------------------------------------------
def list_class_dirs(root: Path) -> dict[str, Path]:
    return {d.name: d for d in sorted(root.iterdir()) if d.is_dir()}


def index_samples(root: Path, code: str, label: int) -> list[tuple[Path, int]]:
    d = root / code
    files = sorted(
        (p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS),
        key=lambda p: str(p).lower(),
    )
    return [(p, label) for p in files]


def load_label_map(path: Path) -> list[dict]:
    """读取 Step 1 的 label_map.json，返回按索引排序的类别表。

    注意文件是 utf-8-sig（带 BOM），必须用 utf-8-sig 读。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    keys = sorted(data.keys(), key=lambda k: int(k))
    return [data[k] for k in keys]


def stratified_split(
    samples: list[tuple[Path, int]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list]:
    """按类别分层切分，保证每类 train/val 比例一致。固定 seed=42。"""
    from sklearn.model_selection import train_test_split

    labels = [lab for _, lab in samples]
    idx = np.arange(len(samples))
    tr_idx, va_idx = train_test_split(
        idx, test_size=val_ratio, stratify=labels, random_state=seed, shuffle=True
    )
    return [samples[i] for i in tr_idx], [samples[i] for i in va_idx]


# --------------------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------------------
def _seed_worker(worker_id: int) -> None:
    """让每个 DataLoader worker 的随机数独立且可复现（训练结果可重复）。"""
    import random as _random

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


def build_dataloaders(
    data_root: str | Path = r"G:\深度学习期末报告\data",
    label_map_path: str | Path = r"G:\深度学习期末报告\outputs\label_map.json",
    img_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    seed: int = 42,
    allow_hflip: bool = False,
    bg_mode: str = "auto",
    norm: str = "imagenet",
    margin: float = 0.12,
    erase_p: float = 0.25,
    erase_mode: str = "white",
    domain_align: bool = False,
    domain_fill: float = 0.88,
    coverage_rand: bool = False,
    pin_memory: bool | None = None,
) -> dict:
    """构建 train / val / test 三个 DataLoader。

    返回 dict：
      train/val/test -> DataLoader
      train_ds/val_ds/test_ds -> Dataset
      classes -> [中文名, ...]（按 label 索引）
      codes   -> [文件夹名, ...]
      stats   -> 样本数等统计
    """
    data_root = Path(data_root)
    train_root = data_root / "train_datasets" / "train"
    test_root = data_root / "test_datasets" / "test"
    for r in (train_root, test_root):
        if not r.is_dir():
            raise FileNotFoundError(f"找不到数据目录: {r}")

    classes = load_label_map(label_map_path)
    codes = [c["code"] for c in classes]
    names = [c["name"] for c in classes]
    train_dirs = list_class_dirs(train_root)
    missing = [c for c in codes if c not in train_dirs]
    if missing:
        raise FileNotFoundError(f"train 目录缺少类别文件夹: {missing}")

    # ---- 训练集样本（切出 val） ----
    all_train: list[tuple[Path, int]] = []
    for label, c in enumerate(classes):
        all_train += index_samples(train_root, c["train_dir"], label)
    train_samples, val_samples = stratified_split(all_train, val_ratio=val_ratio, seed=seed)

    # ---- 测试集样本（全程不参与调参） ----
    test_samples: list[tuple[Path, int]] = []
    for label, c in enumerate(classes):
        test_samples += index_samples(test_root, c["test_dir"], label)

    # ---- transforms ----
    tf_train = build_train_transforms(
        img_size=img_size,
        norm=norm,
        allow_hflip=allow_hflip,
        erase_p=erase_p,
        erase_mode=erase_mode,
    )
    tf_eval = build_eval_transforms(img_size=img_size, norm=norm)

    aug_ops = []
    if domain_align:
        aug_ops.append(RandomFillJitter(base_fill=domain_fill, size=img_size))
        aug_ops.append(RandomMorph(p=0.3))
        if coverage_rand:
            aug_ops.append(InkCoverageFix(lo=0.10, hi=0.28))
        aug_ops.append(RandomBlurJPEG(p=0.3))
        aug_ops.append(RandomSpecks(p=0.3))

    cache_root = data_root / "aligned_cache"
    if not (domain_align and img_size == 128 and cache_root.is_dir()):
        cache_root = None
    if cache_root is not None:
        n_cached = sum(
            1 for p, _ in train_samples
            if (cache_root / p.relative_to(data_root).with_suffix(".png")).exists()
        )
        print(f"[cache] 使用 {cache_root}（train 命中 {n_cached}/{len(train_samples)}）")

    train_ds = CalligraphyDataset(
        train_samples, tf_train, bg_mode=bg_mode, img_size=img_size, margin=margin,
        domain_align=domain_align, fill=domain_fill,
        extra_aug=transforms.Compose(aug_ops) if aug_ops else None,
        cache_root=cache_root, data_root=data_root,
    )
    val_ds = CalligraphyDataset(val_samples, tf_eval, bg_mode=bg_mode, img_size=img_size,
                                margin=margin, domain_align=domain_align, fill=domain_fill,
                                cache_root=cache_root, data_root=data_root)
    test_ds = CalligraphyDataset(test_samples, tf_eval, bg_mode=bg_mode, img_size=img_size,
                                 margin=margin, domain_align=domain_align, fill=domain_fill,
                                 cache_root=cache_root, data_root=data_root)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def _loader(ds, shuffle: bool) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(seed)  # shuffle 顺序也可复现
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
            generator=gen,
        )

    # 类别分布统计
    def _counts(samples: list[tuple[Path, int]]) -> list[int]:
        arr = np.zeros(len(classes), dtype=int)
        for _, lab in samples:
            arr[lab] += 1
        return arr.tolist()

    stats = {
        "img_size": img_size,
        "batch_size": batch_size,
        "val_ratio": val_ratio,
        "seed": seed,
        "allow_hflip": allow_hflip,
        "bg_mode": bg_mode,
        "norm": norm,
        "margin": margin,
        "domain_align": domain_align,
        "domain_fill": domain_fill,
        "erase_p": erase_p,
        "erase_mode": erase_mode,
        "n_classes": len(classes),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "train_per_class": _counts(train_samples),
        "val_per_class": _counts(val_samples),
        "test_per_class": _counts(test_samples),
        "source_total": len(all_train) + len(test_samples),
    }

    return {
        "train": _loader(train_ds, True),
        "val": _loader(val_ds, False),
        "test": _loader(test_ds, False),
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "classes": names,
        "codes": codes,
        "stats": stats,
    }


def denormalize(
    tensor: torch.Tensor,
    norm: str = "imagenet",
) -> torch.Tensor:
    """反归一化到 [0,1]，用于可视化。支持 [C,H,W] 或 [N,C,H,W]。"""
    mean, std = NORM_STATS[norm]
    t = tensor.clone()
    if t.dim() == 3:
        mean_t = torch.tensor(mean).view(-1, 1, 1)
        std_t = torch.tensor(std).view(-1, 1, 1)
    elif t.dim() == 4:
        mean_t = torch.tensor(mean).view(1, -1, 1, 1)
        std_t = torch.tensor(std).view(1, -1, 1, 1)
    else:
        raise ValueError(f"期望 3D/4D 张量，收到 {tuple(t.shape)}")
    return (t * std_t + mean_t).clamp(0.0, 1.0)


if __name__ == "__main__":
    dl = build_dataloaders(batch_size=16, num_workers=0)
    s = dl["stats"]
    print("类别数:", s["n_classes"])
    print(f"train={s['n_train']}  val={s['n_val']}  test={s['n_test']}  (原始 train+test={s['source_total']})")
    print("每类 train:", s["train_per_class"])
    print("每类 val  :", s["val_per_class"])
    x, y = next(iter(dl["train"]))
    print("batch:", tuple(x.shape), x.dtype, "range", float(x.min()), float(x.max()))
    print("labels:", y.tolist())

