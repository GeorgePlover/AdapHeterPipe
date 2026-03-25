import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Linux servers

import numpy as np
import matplotlib.pyplot as plt


def plot_pipeline_training_results(
    methods,
    throughput_k_tokens_per_sec,
    bubble_ratio,
    output_file="pipeline_training_results.pdf",
):
    """
    Plot throughput and average bubble ratio in one figure with two subplots.

    Parameters
    ----------
    methods : list[str]
        Method names.
    throughput_k_tokens_per_sec : list[float]
        Throughput values in K tokens/s.
    bubble_ratio : list[float]
        Average bubble ratio across stages, unit = 1.
        It will be converted to percentage in the plot.
    output_file : str
        Output filename, e.g. PDF or PNG.
    """

    if not (len(methods) == len(throughput_k_tokens_per_sec) == len(bubble_ratio)):
        raise ValueError("All input lists must have the same length.")

    x = np.arange(len(methods))
    bubble_ratio_percent = np.array(bubble_ratio) * 100.0

    # Black-and-white style: grayscale + hatch patterns
    grayscale_colors = ["0.15", "0.28", "0.41", "0.54", "0.67", "0.80", "0.92"]
    hatch_patterns = ["///", "\\\\\\", "xxx", "---", "++", "...", "ooo"]

    # Figure style for single-column paper layout
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Suitable for one column in A4 double-column paper
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    
    # 设置dpi以提升图像清晰度
    

    # -------------------------
    # (a) Throughput
    # -------------------------
    ax = axes[0]
    bars1 = ax.bar(
        x,
        throughput_k_tokens_per_sec,
        width=0.72,
        color=grayscale_colors[:len(methods)],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars1, hatch_patterns):
        bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Throughput (K tokens/s)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="0.78")
    ax.set_axisbelow(True)

    # Increase y-axis upper bound for readability
    ymax1 = max(throughput_k_tokens_per_sec)
    ax.set_ylim(0, ymax1 * 1.22)

    for xi, yi in zip(x, throughput_k_tokens_per_sec):
        ax.text(
            xi,
            yi + ymax1 * 0.025,
            f"{yi:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2
        )

    # Put subplot title below the axis
    ax.text(
        0.5, -0.34,
        "(a) Throughput",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    # -------------------------
    # (b) Bubble ratio
    # -------------------------
    ax = axes[1]
    bars2 = ax.bar(
        x,
        bubble_ratio_percent,
        width=0.72,
        color=grayscale_colors[:len(methods)],
        edgecolor="black",
        linewidth=0.8,
    )
    for bar, hatch in zip(bars2, hatch_patterns):
        bar.set_hatch(hatch)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=28, ha="right")
    ax.set_ylabel("Average bubble ratio (%)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="0.78")
    ax.set_axisbelow(True)

    ymax2 = max(bubble_ratio_percent)
    ax.set_ylim(0, ymax2 * 1.24)

    for xi, yi in zip(x, bubble_ratio_percent):
        ax.text(
            xi,
            yi + ymax2 * 0.025,
            f"{yi:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.2
        )

    ax.text(
        0.5, -0.34,
        "(b) Bubble ratio",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    # Paper-style borders
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # Leave more bottom space for the subplot titles
    fig.subplots_adjust(left=0.10, right=0.995, top=0.96, bottom=0.34, wspace=0.32)

    fig.savefig(output_file, bbox_inches="tight", dpi=1200)
    plt.close(fig)


if __name__ == "__main__":
    methods = ["1F1B", "1F1B-I", "ZB", "ZB-V", "Hexiscale", "Zorse", "AHPipe"]
    throughput_k_tokens_per_sec = [27.0, 29.9, 32.3, 31.6, 27.0, 24.8, 33.3]
    bubble_ratio = [0.363, 0.295, 0.236, 0.255, 0.362, 0.304, 0.151]

    plot_pipeline_training_results(
        methods,
        throughput_k_tokens_per_sec,
        bubble_ratio,
        output_file="pipeline_training_results.png",
    )

    print("Figure saved to pipeline_training_results.png")