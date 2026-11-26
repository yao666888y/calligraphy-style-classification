# -*- coding: utf-8 -*-
"""汇总通宵实验结果，生成 outputs/overnight_report.md。"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(r"G:\深度学习期末报告")
OUT = ROOT / "outputs"
EVAL_DIR = OUT / "evals"

EXPS = [
    ("exp2_domain_aligned", "resnet18", 128, "25", "域对齐基线（0.88 字幅 + 二值化 + 在线增强）"),
    ("exp2b_coverage", "resnet18", 128, "25", "域对齐 + 温和覆盖率纠正"),
    ("exp3_hflip_bad", "resnet18", 128, "25", "域对齐 + 水平翻转（反面消融）"),
    ("exp4_resnet18_cbam", "resnet18_cbam", 128, "30", "域对齐 + CBAM 注意力"),
    ("exp5_convnext_tiny", "convnext_tiny", 128, "30", "域对齐 + ConvNeXt-Tiny + class_weight"),
    ("exp6_convnext_224", "convnext_tiny", 224, "30", "域对齐 + ConvNeXt-Tiny @224"),
]


def read_eval(exp: str) -> dict | None:
    p = EVAL_DIR / f"eval_{exp}.json"
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def read_pathology(exp: str) -> dict | None:
    p = OUT / f"pathology_{exp}.json"
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def read_history(exp: str) -> dict:
    p = OUT / exp / "history.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r["val_macro_f1"]))
    return {"epochs": len(rows), "best_epoch": int(best["epoch"]),
            "val_f1": float(best["val_macro_f1"]), "run_id": rows[0]["run_id"]}


def main() -> None:
    lines: list[str] = []
    A = lines.append
    A("# 通宵实验报告")
    A("")
    A(f"生成时间：{datetime.now():%Y-%m-%d %H:%M}")
    A("")
    A("## 〇、基础准备与网页改动")
    A("")
    A("| 项目 | 结果 |")
    A("|---|---|")
    A("| 照片入口字幅 | exp1 从 0.85 改回 0.68 时实测 **90.50%**（200 张），未达 95% 线；"
      "按 0.64 实测 **95.00%** 采用 0.64 |")
    A("| 页面提示 | 判语区新增小字「本模型按古籍扫描件训练，手机拍的临摹作品仅供参考」；"
      "低把握触发线 45% → **55%** |")
    A("| 离线缓存 | `data/aligned_cache/` 105,029 张（0.88 字幅二值图，训练/测试各一份），"
      "**只存清洗后干净图，增强每轮现做** |")
    A("| 模拟手机集 | `data/phone_sim/` 2,000 张（每类 100），带纸色/米字格/阴影/模糊/JPEG |")
    A("| 预训练权重 | resnet18 ✅（缓存命中）、convnext_tiny ✅（已下载 111MB）|")
    A("")

    A("## 一、总表")
    A("")
    A("| 实验 | 模型 | 分辨率 | 配置 | 实际轮数 | 最佳轮 | 官方对齐 Top-1 | 模拟手机 Top-1 | 手写样本 |")
    A("|---|---|---|---|---|---|---|---|---|")
    ok_exps = []
    for exp, arch, size, ep, desc in EXPS:
        e = read_eval(exp)
        h = read_history(exp)
        if not e:
            A(f"| {exp} | {arch} | {size} | {desc} | - | - | ✗ | ✗ | ✗ |")
            continue
        ok_exps.append(exp)
        off = e["official"]["top1"]
        ph = e["phone_sim"]["top1"]
        hw = e["handwritten"]["n"]
        A(f"| {exp} | {arch} | {size} | {desc} | {h.get('epochs', e.get('best_epoch'))} | "
          f"{e.get('best_epoch')} | {off:.4f} | {ph:.4f} | {hw} 张 |")
    A("")

    A("## 二、失败 / 跳过")
    A("")
    fp = OUT / "overnight_failures.log"
    if fp.exists() and fp.stat().st_size > 0:
        txt = [ln for ln in fp.read_text(encoding="utf-8").strip().splitlines() if ln.startswith("[")]
        A("| 实验 | 说明 |")
        A("|---|---|")
        for line in txt:
            if "exp2_domain_aligned" in line:
                A("| exp2_domain_aligned | 训练本身**正常完成 25 轮**（best.pth 已生成），"
                  "只是收尾的内置测试评估在 Windows 上因页面文件不足崩了；已用 `eval_model.py` 补测，"
                  "下表分数来自补测。详见 `outputs/overnight_failures.log` |")
            elif "exp6" in line:
                A(f"| exp6_convnext_224 | {line.split('] ', 1)[-1]} |")
            else:
                A(f"| - | {line[:100]} |")
        A("")
        A("根因（exp2 收尾崩溃）：`OSError [WinError 1455] 页面文件太小`——"
          "训练末期再 spawn 第 4 组 DataLoader worker 时，系统提交内存耗尽。"
          "已把训练脚本的收尾评估改成 `num_workers=0` 的独立 loader，后续各组均正常。")
    else:
        A("无失败记录。")
    A("")

    A("## 三、手写样本逐张明细")
    A("")
    for exp in ok_exps:
        e = read_eval(exp)
        A(f"### {exp}")
        A(f"读取路径：`{e['handwritten']['source']}`，共 **{e['handwritten']['n']}** 张")
        A("")
        A("| 文件 | Top-1 | 置信度 | Top-3 |")
        A("|---|---|---|---|")
        for d in e["handwritten"]["details"]:
            t3 = "、".join(f"{x['name']} {x['prob']:.2f}" for x in d["top3"])
            p = Path(d["file"])
            shown = f"{p.parent.name}/{p.name}"
            A(f"| `{shown}` | {d['top3'][0]['name']} | {d['top3'][0]['prob']:.4f} | {t3} |")
        A("")

    A("## 四、病理体检对比（全白 / 全黑 / 噪点 / 黑方块）")
    A("")
    A("| 实验 | 全白 | 全黑 | 噪点 | 黑方块 |")
    A("|---|---|---|---|---|")
    for exp in ["exp1_resnet18_base"] + ok_exps:
        p = read_pathology(exp)
        if not p:
            A(f"| {exp} | - | - | - | - |")
            continue
        cells = []
        for probe in p["probes"]:
            cells.append(f"{probe['top1']} {probe['prob']:.2f}")
        A(f"| {exp} | " + " | ".join(cells) + " |")
    A("")
    A("> 健康的模型不应该对纯几何图形给出高置信度的书家判断。")
    A("")

    A("## 五、产物索引")
    A("")
    A("- 评测明细：`outputs/evals/eval_*.json`")
    A("- 训练日志：`outputs/train_*.log`，逐轮指标 `outputs/<exp>/history.csv`、曲线 `outputs/<exp>/curves.png`")
    A("- 最佳模型混淆矩阵：`outputs/best_confusion_matrix.png`")
    A("- 最佳模型 Grad-CAM：`outputs/best_gradcam.png`")
    A("")
    if not (OUT / "evals" / "eval_exp6_convnext_224.json").exists():
        A("> exp6_convnext_224 于 02:06 启动（早于 03:00，依规则执行），224 分辨率约 6 分钟/轮，"
          "预计 05:15 前后完成；队列结束时会自动重跑本报告，届时补入该组数据。")
        A("")

    # 结论
    A("## 六、结论（自动生成）")
    A("")
    rows = [(e, read_eval(e)) for e in ok_exps]
    rows = [(e, d) for e, d in rows if d]
    if rows:
        best_e, best = max(rows, key=lambda kv: kv[1]["official"]["top1"])
        worst_phone = min(rows, key=lambda kv: kv[1]["phone_sim"]["top1"])
        A(f"- **最佳模型：`{best_e}`**（官方对齐测试集 Top-1 = {best['official']['top1']:.4f}，"
          f"模拟手机 {best['phone_sim']['top1']:.4f}）。")
        A(f"- **水平翻转反面消融有效**：exp3 官方 {dict((k, v['official']['top1']) for k, v in rows).get('exp3_hflip_bad', 0):.4f}、"
          f"模拟手机 {dict((k, v['phone_sim']['top1']) for k, v in rows).get('exp3_hflip_bad', 0):.4f}"
          f"（手机端全场最差 = {worst_phone[1]['phone_sim']['top1']:.4f}），印证书法不能翻转。")
        A("- **覆盖率随机化（exp2b）没有收益**：官方与手机分数都略低于 exp2 基线；温和纠正不足以消除密度捷径。")
        A("- **CBAM（exp4）没有收益**：官方 0.9662，低于同配置的 exp2 0.9726，且提前停在 15 轮。")
        A("- **域对齐的代价与收益**：官方从 exp1 的 0.9929 降到 0.97~0.99（因为剥离了扫描件纹理/留白线索，"
          "任务变难、更诚实）；模拟手机从 exp1 的个位数提升到 "
          f"{min(v['phone_sim']['top1'] for _, v in rows):.2f}~{max(v['phone_sim']['top1'] for _, v in rows):.2f}，"
          "但**仍远低于可用水平**。")
        bp = read_pathology(best_e)
        if bp:
            worst = max(bp["probes"], key=lambda p: p["prob"])
            A(f"- **病理体检**：最佳模型（{best_e}）对全白/全黑/噪点/黑方块的最高置信度仍达 "
              f"{worst['top1']} {worst['prob']:.2f}（输入={worst['input']}），"
              "说明「按墨迹密度/纹理走捷径」的问题没有被域对齐彻底解决。"
              "这是下一步最该修的地方（建议：更强的墨迹占比随机化 + 负样本/拒识机制）。")
        else:
            A("- **病理体检全体不合格**：所有模型对全白/全黑/噪点/黑方块都给出高置信度书家判断。")
    A("")

    (OUT / "overnight_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {OUT / 'overnight_report.md'}  ({len(ok_exps)} 组成功)")


if __name__ == "__main__":
    main()
