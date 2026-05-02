#!/usr/bin/env python3
"""
独立甘特图绘制脚本
读取 data/ 目录下的任务数据 JSON 文件，生成流水线调度甘特图。

用法: python plot_gantt.py

依赖: matplotlib (pip install matplotlib)
"""

import json
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.patheffects as pe
from pathlib import Path

# ============================================================================
# 可调参数区 —— 编辑老师可直接修改以下参数调整输出效果
# ============================================================================

# --------------------------- 文件路径 ---------------------------
DATA_DIR = Path(__file__).parent / "data"          # JSON 数据目录
OUTPUT_DIR = Path(__file__).parent / "output"       # 输出目录

# --------------------------- 图尺寸与分辨率 ---------------------------
FIGURE_WIDTH = 12         # 图片宽度 (英寸)
FIGURE_HEIGHT = 4         # 图片高度 (英寸)
DPI = 600                 # 输出分辨率

# --------------------------- 颜色 ---------------------------
COLOR_FORWARD = "#5CA9C8"     # 前向传播 (F)
COLOR_BACKWARD = "#5DB558"    # 反向传播 (B)
COLOR_WEIGHT = "#C77337"      # 权重梯度计算 (W)
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1.5
BAR_ALPHA = 0.8
BAR_HATCH = ""               # 条形填充纹理: '', '/', '\\', 'x', etc.

# --------------------------- 字号 (磅) ---------------------------
FONT_SIZE_BAR_LABEL = 16     # 条形内 microbatch 编号字号
FONT_SIZE_XLABEL = 22        # X 轴标签字号
FONT_SIZE_YTICK = 22         # Y 轴刻度字号
FONT_SIZE_LEGEND = 25        # 图例字号

# --------------------------- 文字样式 ---------------------------
FONT_WEIGHT_LABEL = "bold"   # 条形内标注粗细
FONT_WEIGHT_AXIS = "bold"    # 轴标签/刻度粗细
SHOW_MICROBATCH_IDS = True   # 是否在条形内显示 microbatch 编号
BOLD_STROKE_WIDTH = 1.2      # 无真粗体时的描边模拟宽度 (0 为禁用)

# --------------------------- 中文字体 (宋体优先) ---------------------------
CN_FONT_CANDIDATES = [
    "SimSun",               # Windows 宋体
    "Songti SC",            # macOS 宋体
    "Noto Serif CJK SC",    # Linux 思源宋体
    "WenQuanYi Zen Hei",    # Linux 文泉驿正黑 (降级)
    "Droid Sans Fallback",  # Android/Droid
    "DejaVu Sans",          # 终极降级
]
# --------------------------- 英文字体 ---------------------------
EN_FONT_CANDIDATES = [
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
    "serif",
]

# --------------------------- 图表映射 (JSON文件名 -> 输出PNG名) ---------------------------
CHART_CONFIGS = {
    "GPipe_draw": {
        "json_file": "GPipe_draw_task_data.json",
        "output_png": "图1.png",
        "label": "GPipe",
    },
    "1F1B_draw": {
        "json_file": "1F1B_draw_task_data.json",
        "output_png": "图2.png",
        "label": "1F1B",
    },
    "Interleaved_1F1B_draw": {
        "json_file": "Interleaved_1F1B_draw_task_data.json",
        "output_png": "图3-b.png",
        "label": "Interleaved 1F1B",
    },
    "ZB_draw": {
        "json_file": "ZB_draw_task_data.json",
        "output_png": "图4.png",
        "label": "ZB",
    },
    "ZB_Vshape_draw": {
        "json_file": "ZB_Vshape_draw_task_data.json",
        "output_png": "图5.png",
        "label": "ZB-V",
    },
}

# ============================================================================
# 字体检测与设置
# ============================================================================

def _find_font(candidates, fallback="sans-serif"):
    """在系统中查找第一个可用的字体，返回字体名称。"""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    warnings.warn(f"未找到候选字体 {candidates}，使用降级字体 '{fallback}'")
    return fallback

def _has_true_bold(font_name):
    """检查字体是否有原生 Bold 字重 (weight>=600)。"""
    for f in fm.fontManager.ttflist:
        if f.name == font_name and f.weight >= 600:
            return True
    return False

CN_FONT = _find_font(CN_FONT_CANDIDATES)
EN_FONT = _find_font(EN_FONT_CANDIDATES)

plt.rcParams["font.sans-serif"] = [CN_FONT]
plt.rcParams["font.serif"] = [EN_FONT]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

try:
    plt.rcParams["font.fallback"] = ["serif"]
except KeyError:
    pass

# 粗体策略: 有原生 Bold 就用, 否则用 path_effects 描边模拟
TRUE_BOLD = _has_true_bold(CN_FONT)
if TRUE_BOLD:
    print(f"[字体] 中文: {CN_FONT} (原生粗体), 英文: {EN_FONT}")
else:
    if BOLD_STROKE_WIDTH > 0:
        print(f"[字体] 中文: {CN_FONT} (粗体通过描边模拟, width={BOLD_STROKE_WIDTH}), 英文: {EN_FONT}")
    else:
        print(f"[字体] 中文: {CN_FONT} (无粗体, 将使用常规字重), 英文: {EN_FONT}")

def _make_bold_stroke(foreground="black"):
    """根据前景色生成描边效果 (用于 path_effects 模拟粗体)"""
    if BOLD_STROKE_WIDTH <= 0 or TRUE_BOLD:
        return []
    return [pe.withStroke(linewidth=BOLD_STROKE_WIDTH, foreground=foreground)]

_BOLD_EFFECT = _make_bold_stroke("black")
_BOLD_KWARGS = {"path_effects": _BOLD_EFFECT} if _BOLD_EFFECT else {}
_BOLD_WEIGHT = FONT_WEIGHT_AXIS if TRUE_BOLD else "normal"
_BOLD_WEIGHT_LABEL = FONT_WEIGHT_LABEL if TRUE_BOLD else "normal"

# ============================================================================
# 甘特图绘制函数
# ============================================================================

def generate_gantt_chart(
    tasks,
    filepath,
    label="",
    no_W=False,
    stage_half_white_text=False,
):
    """
    绘制甘特图

    参数:
        tasks:           任务列表, 每个元素含 worker_id, stage_id, microbatch_id,
                         task_type, start_time, end_time, duration
        filepath:        输出文件路径
        label:           图表标识 (用于日志)
        no_W:            是否无 W 任务 (影响图例)
        stage_half_white_text:  是否后半阶段用白色字体
    """
    stage_cnt = max(t["stage_id"] for t in tasks) + 1 if tasks else 1

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    base_color_map = {
        "F": COLOR_FORWARD,
        "B": COLOR_BACKWARD,
        "W": COLOR_WEIGHT,
    }

    # 构建颜色渐变映射
    color_gradient = {}
    for task_type, base_color in base_color_map.items():
        base_rgb = mcolors.to_rgb(base_color)
        light_color = tuple(c * 0.4 + 0.6 for c in base_rgb)
        color_gradient[task_type] = mcolors.LinearSegmentedColormap.from_list(
            f"{task_type}_gradient", [light_color, base_rgb]
        )

    max_time = max(t["end_time"] for t in tasks) if tasks else 1

    for task in tasks:
        duration = task["duration"]
        if duration <= 0:
            continue

        stage_id = task["stage_id"]
        task_type = task["task_type"]

        # 颜色强度
        if stage_half_white_text and stage_id >= stage_cnt // 2:
            color_intensity = 1.0
            font_color = "white"
        else:
            color_intensity = 1.0
            font_color = "black"

        task_color = color_gradient[task_type](color_intensity)

        ax.barh(
            y=task["worker_id"],
            width=duration,
            left=task["start_time"],
            height=1.0,
            color=task_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
            hatch=BAR_HATCH,
            alpha=BAR_ALPHA,
        )

        if SHOW_MICROBATCH_IDS:
            _bar_stroke = _make_bold_stroke(font_color)
            ax.text(
                task["start_time"] + duration / 2,
                task["worker_id"],
                str(task["microbatch_id"]),
                ha="center",
                va="center",
                fontsize=FONT_SIZE_BAR_LABEL,
                weight=_BOLD_WEIGHT_LABEL,
                color=font_color,
                path_effects=_bar_stroke,
            )

    # X 轴标签 (右对齐)
    ax.set_xlabel(
        "时间",
        fontdict={"size": FONT_SIZE_XLABEL, "weight": _BOLD_WEIGHT},
        ha="right",
        x=0.95,
        **_BOLD_KWARGS,
    )

    # Y 轴标签
    if tasks:
        worker_ids = sorted(set(t["worker_id"] for t in tasks), reverse=True)
        y_labels = [f"GPU-{wid}" for wid in worker_ids]
        ax.set_yticks(range(len(worker_ids)))
        ax.set_yticklabels(y_labels)

    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=FONT_SIZE_YTICK)
    for lbl in ax.get_yticklabels():
        lbl.set_weight(_BOLD_WEIGHT)
        if _BOLD_KWARGS:
            lbl.set_path_effects(_BOLD_KWARGS.get("path_effects", []))

    # 边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["bottom"].set_capstyle("projecting")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticklabels([])

    # 图例
    ncol = 2
    f_patch = mpatches.Patch(
        facecolor=COLOR_FORWARD, label="前向传播", edgecolor=BAR_EDGE_COLOR
    )
    b_patch = mpatches.Patch(
        facecolor=COLOR_BACKWARD, label="反向传播", edgecolor=BAR_EDGE_COLOR
    )
    handles = [f_patch, b_patch]
    if not no_W:
        w_patch = mpatches.Patch(
            facecolor=COLOR_WEIGHT, label="权重梯度计算", edgecolor=BAR_EDGE_COLOR
        )
        handles.append(w_patch)
        ncol = 3

    legend = ax.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.05),
        fontsize=FONT_SIZE_LEGEND,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_weight(_BOLD_WEIGHT)
        if _BOLD_KWARGS:
            text.set_path_effects(_BOLD_KWARGS.get("path_effects", []))

    plt.tight_layout()
    plt.savefig(filepath, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"[生成] {label} -> {filepath}")


# ============================================================================
# 主入口
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for key, cfg in CHART_CONFIGS.items():
        json_path = DATA_DIR / cfg["json_file"]
        if not json_path.exists():
            print(f"[跳过] 未找到数据文件: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            tasks = json.load(f)

        # 判断 no_W (数据中是否有 W 类型任务)
        has_W = any(t["task_type"] == "W" for t in tasks)
        no_W = not has_W

        # 判断是否后半阶段白字 (按原始逻辑: stage>=8 的 Interleaved/ZB_Vshape)
        stage_cnt = max(t["stage_id"] for t in tasks) + 1 if tasks else 1
        stage_half_white = stage_cnt > 4  # 超过4个阶段视为交错/V形模式

        output_path = OUTPUT_DIR / cfg["output_png"]
        generate_gantt_chart(
            tasks,
            str(output_path),
            label=cfg["label"],
            no_W=no_W,
            stage_half_white_text=stage_half_white,
        )

    print(f"\n全部完成！输出目录: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
