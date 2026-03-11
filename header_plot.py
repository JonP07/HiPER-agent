import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# Data
# -----------------------------
methods = ["Ours", "PPO", "GRPO", "GiGPO"]

alfworld_1_5b = [95.3, 68.2, 71.1, 86.7]
webshop_1_5b  = [71.4, 51.5, 56.8, 67.4]
alfworld_7b   = [97.4, 82.8, 85.4, 90.8]
webshop_7b    = [83.3, 68.7, 66.1, 75.2]

colors = {
    "Ours":  "#E6D9F6",
    "PPO":   "#A9A9A9",
    "GRPO":  "#D2C8BE",
    "GiGPO": "#F2F3C9",
}

plt.rcParams.update({"font.family": "DejaVu Serif"})

# -----------------------------
# Plot helper
# -----------------------------
def plot_panel(ax, title, values, ylim, yticks, show_ylabel=False, bar_w=0.4):
    vals = np.asarray(values, dtype=float)
    keep = ~np.isnan(vals)
    m_used = [m for m, k in zip(methods, keep) if k]
    v_used = vals[keep]

    x = np.arange(len(m_used))

    ax.bar(
        x, v_used, width=bar_w,
        color=[colors[m] for m in m_used],
        edgecolor="#6B6B6B", linewidth=1.5, zorder=3
    )

    for xi, yi in zip(x, v_used):
        ax.text(
            xi,
            yi + (ylim[1] - ylim[0]) * 0.02,
            f"{yi:.2f}",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", zorder=4
        )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=10)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)

    ax.grid(axis="y", linestyle="--", linewidth=1.2, alpha=0.35, zorder=0)
    ax.set_xticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.tick_params(axis="y", labelsize=14)

    if show_ylabel:
        ax.set_ylabel("Accuracy (%)", fontsize=18, fontweight="bold")
    else:
        ax.set_ylabel("")

# -----------------------------
# 1x4 layout
# -----------------------------
BAR_WIDTH = 0.4

fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
ax1, ax2, ax3, ax4 = axes

plot_panel(
    ax1, "ALFWorld-1.5B", alfworld_1_5b,
    ylim=(50, 100), yticks=np.arange(50, 101, 10),
    show_ylabel=True, bar_w=BAR_WIDTH
)
plot_panel(
    ax2, "WebShop-1.5B", webshop_1_5b,
    ylim=(40, 75), yticks=np.arange(40, 76, 5),
    show_ylabel=False, bar_w=BAR_WIDTH
)
plot_panel(
    ax3, "ALFWorld-7B", alfworld_7b,
    ylim=(50, 100), yticks=np.arange(50, 101, 10),
    show_ylabel=False, bar_w=BAR_WIDTH
)
plot_panel(
    ax4, "WebShop-7B", webshop_7b,
    ylim=(55, 85), yticks=np.arange(55, 86, 5),
    show_ylabel=False, bar_w=BAR_WIDTH
)

# Legend across the bottom (4 methods => ncol=4)
legend_handles = [
    Patch(facecolor=colors[m], edgecolor="#6B6B6B", linewidth=1.5, label=m)
    for m in methods
]
leg = fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    bbox_to_anchor=(0.5, -0.02),
    fontsize=16,
)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.6)

plt.tight_layout(rect=(0, 0.12, 1, 1))
# plt.show()
plt.savefig("/projects/standard/mhong/peng0504/HGAE-Agent/verl-agent/header_plot.pdf", dpi=300, bbox_inches="tight")