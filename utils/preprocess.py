# -*- coding: utf-8 -*-
"""训练与推理共用的书法单字归一化。

设计目标：让「网页上传的照片」和「训练集里的扫描件」走**同一条**处理链，
输出同一套几何与灰度规格，消除训练/推理分布分歧。

流程
----
1. 转灰度（PIL，中文路径安全）
2. 纸面归一化：形态学闭运算估计纸面亮度，逐像素相除 → 阴影/渐变光照被洗成纯白
3. 自适应阈值（窗口按估计笔画宽度自适应，避免粗笔画被掏空）
4. 连通域清理：
   * 面积 < min_component_ratio（默认 0.05%）的孤立小噪点 → 丢弃（米字格残点不会撑大外接框）
   * 空心页框（外接框覆盖 ≥85% 且填充率 <25%）→ 丢弃（整页黑框）
5. 按剩余墨迹求**紧致外接框**
6. 等比缩放，使汉字长边占画布 fill（默认 0.88），居中贴在纯白正方形上
7. binarize=True 时输出纯黑/纯白（剥离扫描灰度纹理）；False 时保留笔画软边灰度

注意：本函数不做任何旋转/翻转/遮挡，那些属于训练增强，见 utils/dataset.py。
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

WHITE = 255
BLACK = 0


@dataclass
class NormalizeInfo:
    input_size: tuple[int, int]
    stroke_px: float
    window: int
    ink_ratio: float
    bbox: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    fill: float
    scale: float
    n_components_dropped: int
    frame_dropped: bool
    binarized: bool
    dark_bg: bool = False
    ink_src_ratio: float = 0.0      # 紧致框内「灰度<128」的占比（清洗前）
    ink_mask_ratio: float = 0.0     # 紧致框内「掩膜」的占比（清洗后）
    ink_final_ratio: float = 0.0    # 最终 128/224 画布上的墨迹占比
    glyph_fill_actual: float = 0.0  # 最终字幅长边实际占画布比例


def _border_median(g: np.ndarray) -> float:
    """四边 6% 条带的灰度中位数，用来判断是否黑底（拓片）。"""
    h, w = g.shape
    b = max(2, int(round(min(h, w) * 0.06)))
    edge = np.concatenate([g[:b].ravel(), g[-b:].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
    return float(np.median(edge))


def _stroke_width(g: np.ndarray) -> float:
    """用 Otsu 掩膜 + 距离变换估计平均笔画宽度（像素）。"""
    if float(g.std()) < 0.5:          # 纯色图（全白/全黑）没有笔画
        return 6.0
    _, tmp = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(tmp, cv2.DIST_L2, 5)
    vals = dist[np.isfinite(dist) & (dist > 0)]
    if not vals.size:
        return 6.0
    sw = float(2.0 * vals.mean())
    return sw if np.isfinite(sw) and sw > 0 else 6.0


def fill_small_holes(img: Image.Image, ratio: float = 0.0015) -> tuple[Image.Image, int]:
    """填充「被墨包围、面积 < ratio×整图」的白色孔洞（碑刻石花 / 飞白白点）。

    只填小块封闭白区；汉字正常的封闭留白（口、日、田里的空腔）面积远大于阈值，
    所以不会被误填。返回 (处理后的图, 填充个数)。
    """
    a = np.asarray(img.convert("L"))
    ink = a < 128
    inv = (~ink).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num <= 1:
        return img, 0
    h, w = a.shape
    out = ink.copy()
    filled = 0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area >= ratio * h * w:
            continue
        if x <= 0 or y <= 0 or x + ww >= w or y + hh >= h:
            continue                      # 触边的是背景，不动
        out[labels == i] = True
        filled += 1
    if filled == 0:
        return img, 0
    return Image.fromarray(np.where(out, 0, 255).astype(np.uint8)).convert("RGB"), filled


def legacy_dataset_style(
    img: Image.Image,
    size: int = 128,
    margin: float = 0.12,
) -> Image.Image:
    """第 1 组（exp1）训练时用的取景方式，原样复刻，供其推理对齐用。

    与 normalize_calligraphy_image 的区别：**不按墨迹紧致裁切**，
    保留原图自带的留白与构图，只做「底色统一 → 补 12% 白边 → 缩放」。
    exp1 就是在这个分布上训练的，换成紧致裁切会掉 30 个百分点。
    """
    from utils.dataset import pad_to_square, unify_background

    rgb = img.convert("RGB")
    out, _ = unify_background(rgb, "auto")
    out = pad_to_square(out, margin=margin)
    return out.resize((size, size), Image.LANCZOS)


def normalize_calligraphy_image(
    img: Image.Image,
    size: int = 128,
    fill: float = 0.88,
    binarize: bool = True,
    mask_background: bool | str = True,
    min_component_ratio: float = 0.0005,
    border_trim_ratio: float = 0.002,
    min_border_side: int = 200,
    photo_min_pixels: int = 150_000,
    return_info: bool = False,
):
    """把一张书法单字图归一化成 size×size 的标准输入。

    img                 : PIL 图（手机照片 / 扫描件 / 64px 数据集图都行）
    size                : 输出边长
    fill                : 汉字长边占画布比例（0~1）
    binarize            : True → 纯黑纯白；False → 保留笔画软边灰度
    mask_background     : True → 掩膜外强制刷白；False → 保留原灰度；
                          "auto" → 只在「照片级大图」或「检测到网格/页框/大量碎屑」时刷白
                          （小字帖图刷白会掉点，但网格虚线又必须刷掉，所以默认 auto）
    min_component_ratio : 小于「整图面积 × 该比例」的连通域视为噪点丢弃
    border_trim_ratio   : 处理前先裁掉的边缘比例（只对 >=min_border_side 的大图生效）
    """
    rgb = img.convert("RGB")
    g = np.asarray(rgb.convert("L"), dtype=np.uint8)
    h0, w0 = g.shape

    # ---- 1) 去掉最外圈压缩伪影（大图才做；64px 小图裁 1px 就会改变取景）----
    if border_trim_ratio > 0 and min(g.shape) >= min_border_side:
        t = max(1, int(round(border_trim_ratio * min(g.shape))))
        if g.shape[0] > 2 * t + 8 and g.shape[1] > 2 * t + 8:
            g = g[t:-t, t:-t]

    h, w = g.shape
    if min(h, w) < 8:
        raise ValueError("图片尺寸太小")

    # ---- 1.5) 黑底白字（拓片）→ 先反色 ----
    # 拓片走自适应阈值会被「掏空」成细灰线：反色后改用全局 Otsu + 闭运算填实
    dark_bg = _border_median(g) < 85
    if dark_bg:
        g = (255 - g).astype(np.uint8)

    # ---- 2) 纸面归一化（闭运算估底色）----
    sw = _stroke_width(g)
    k = int(max(11, min(round(4.5 * sw) | 1, (min(h, w) - 1) | 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    paper = np.maximum(cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel).astype(np.float32), 1.0)
    norm = np.clip(g.astype(np.float32) / paper * 255.0, 0, 255).astype(np.uint8)

    # ---- 3) 二值化：拓片走「全局 Otsu + 闭运算填实」，普通照片走自适应阈值 ----
    if dark_bg:
        _, binimg = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ink = binimg < 128
        # 闭运算把笔画内部的白色麻点填实（拓片纸纹会造成空心）
        kc = int(max(3, min(11, round(sw * 0.8))) | 1)
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))) > 0
    else:
        binimg = cv2.adaptiveThreshold(
            norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, 12
        )
        binimg = cv2.medianBlur(binimg, 3)
        ink = binimg < 128
        if ink.mean() > 0.5:                   # 兜底：万一还是黑底
            ink = ~ink
    if not ink.any():
        raise ValueError("图里没找到墨迹")

    # ---- 3.5) 照片级：先做 3×3 开运算，直接抹掉 1~2px 的格线碎段 ----
    # （笔画在照片里通常 >=5px，开运算只削掉一层毛边，不影响字形）
    if dark_bg is False and float(h * w) >= photo_min_pixels:
        ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((3, 3), np.uint8)) > 0
        if not ink.any():
            raise ValueError("墨迹太小")

    # ---- 4) 连通域清理 ----
    photo_like = float(h * w) >= photo_min_pixels
    num, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    dropped, frame_dropped = 0, False
    if num > 1:
        min_area = max(1.0, min_component_ratio * float(h * w))
        thin_t = max(3.0, 0.012 * min(h, w))       # 细线阈值（按图像尺寸缩放）
        thin_area = 0.005 * float(h * w)
        main_area = float(stats[1:, cv2.CC_STAT_AREA].max())
        # 细小长条（格线残段）：只占主体 2% 以下、极细、长宽比 >=5
        line_thick = max(2.5, 0.008 * min(h, w))
        # 去格线规则只对「照片级」图生效（约 >=141×141）：64px 字帖小图没有格线，
        # 强行套用会误删细笔画（实测准确率 95%→89%）
        grid_cleanup = float(h * w) >= 20_000
        keep, frames = [], []
        for i in range(1, num):
            x, y, ww, hh, a = stats[i]
            thick = a / max(1, max(ww, hh))
            aspect = max(ww, hh) / max(1, min(ww, hh))
            if ww >= 0.85 * w and hh >= 0.85 * h and a / max(1, ww * hh) < 0.25:
                frames.append(i)               # 空心页框
            elif a < min_area:
                dropped += 1                    # 孤立小噪点（米字格残点等）
            elif photo_like and thick < thin_t and a < thin_area:
                dropped += 1                    # 照片里的细长虚线（田字格/米字格）
            elif grid_cleanup and a < 0.02 * main_area and thick < line_thick and aspect >= 5:
                dropped += 1                    # 任何尺寸下的格线残段（不依赖 photo_like）
            elif grid_cleanup and thick < 0.04 * min(h, w) and aspect >= 4 and a < 0.10 * main_area:
                dropped += 1                    # 短横/短竖碎段（米字格虚线残段）
            elif grid_cleanup and (max(ww, hh) >= 0.8 * max(w, h)) and thick < 0.03 * min(h, w):
                dropped += 1                    # 贯穿半个画面的细长条（界格横/竖线）
            else:
                keep.append(i)
        if frames and keep:
            frame_dropped = True
            dropped += len(frames)
        if keep:
            ink = np.isin(labels, keep)
        elif frames:                            # 只剩框 → 退回全部
            ink = np.isin(labels, list(range(1, num)))
        if not ink.any():
            raise ValueError("墨迹太小")

    # ---- 5) 紧致外接框 ----
    ys, xs = np.where(ink)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    # 是否把掩膜外刷白：照片级大图 / 检测到页框 / 丢弃过 3 个以上碎屑（虚线网格）
    if mask_background == "auto":
        do_mask = photo_like or frame_dropped or dropped >= 3
    else:
        do_mask = bool(mask_background)

    glyph = (
        np.where(ink, BLACK, WHITE).astype(np.uint8)
        if binarize
        else (
            np.where(ink, norm, WHITE).astype(np.uint8)
            if do_mask
            else norm.astype(np.uint8)
        )
    )
    if do_mask and not binarize:
        # 掩膜外已刷白，但少量淡灰残线（虚线网格、纸纹）可能被判进掩膜内部，
        # 这里把「浅灰」统一刷白，只保留真正的墨色，避免底上留下脏点
        glyph[glyph > 200] = WHITE
    crop = glyph[y0 : y1 + 1, x0 : x1 + 1]
    ch, cw = crop.shape

    # 自检：清洗前后墨迹占比（同一个紧致框内比较，排除缩放/补白的影响）
    # 「清洗前」用偏宽松的绝对阈值(<160)统计灰度墨迹，和掩膜口径可比，
    # 否则会把抗锯齿过渡带算成「被阈值啃掉」
    sub_norm = norm[y0 : y1 + 1, x0 : x1 + 1]
    sub_ink = ink[y0 : y1 + 1, x0 : x1 + 1]
    ink_src_ratio = float((sub_norm < 160).mean())      # 清洗前：灰度意义上的墨
    ink_mask_ratio = float(sub_ink.mean())              # 清洗后：掩膜墨

    # ---- 6) 长边占到 fill，居中贴白 ----
    scale = (fill * size) / max(ch, cw)
    new_w, new_h = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = Image.fromarray(crop, mode="L").resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (size, size), WHITE)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    out = canvas.convert("RGB")

    # 最终画布上的实际指标
    arr_out = np.asarray(out.convert("L"))
    ink_final_ratio = float((arr_out < 128).mean())
    ys2, xs2 = np.where(arr_out < 128)
    glyph_fill_actual = (max(int(xs2.max() - xs2.min() + 1), int(ys2.max() - ys2.min() + 1)) / size
                         if ys2.size else 0.0)

    if not return_info:
        return out
    info = NormalizeInfo(
        input_size=(w0, h0),
        stroke_px=round(sw, 1),
        window=k,
        ink_ratio=round(float(ink.mean()), 4),
        bbox=(x0, y0, x1, y1),
        crop_size=(cw, ch),
        fill=fill,
        scale=round(scale, 4),
        n_components_dropped=dropped,
        frame_dropped=frame_dropped,
        binarized=binarize,
        dark_bg=dark_bg,
        ink_src_ratio=round(ink_src_ratio, 4),
        ink_mask_ratio=round(ink_mask_ratio, 4),
        ink_final_ratio=round(ink_final_ratio, 4),
        glyph_fill_actual=round(glyph_fill_actual, 4),
    )
    return out, info
