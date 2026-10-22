"""Regenerate the chordal explainer's SVG figures.

Run with Python, NumPy and Matplotlib installed:
    python docs/_scripts/chordal_figures.py

These illustrate patterns and analytic counts, not measured timings.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np

OUTPUT = Path(__file__).resolve().parents[1] / "_static" / "figures"
BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#d97706"
RED = "#b91c1c"
INK = "#172033"
MUTED = "#526177"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "moreau-chordal-explainer",
        "savefig.facecolor": "white",
    }
)


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / name
    fig.savefig(destination, bbox_inches="tight", metadata={"Date": None})
    destination.write_text(
        "\n".join(line.rstrip() for line in destination.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def graph(ax, positions, edges, *, fill=(), separator=()):
    for i, j in edges:
        ax.plot(*np.array([positions[i], positions[j]]).T, color=MUTED, lw=2, zorder=1)
    for i, j in fill:
        ax.plot(
            *np.array([positions[i], positions[j]]).T, color=AMBER, lw=3, linestyle="--", zorder=2
        )
    for i, (x, y) in positions.items():
        color = AMBER if i in separator else BLUE
        ax.add_patch(Circle((x, y), 0.12, facecolor=color, edgecolor="white", lw=2, zorder=3))
        ax.text(x, y, str(i + 1), color="white", ha="center", va="center", weight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")


def cycle_figure():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), layout="constrained")
    positions = {0: (-0.8, 0.8), 1: (0.8, 0.8), 2: (0.8, -0.8), 3: (-0.8, -0.8)}
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph(axes[0], positions, edges)
    axes[0].set_title("1. Sparse four-cycle", pad=10)
    graph(axes[1], positions, edges, fill=[(0, 2)], separator={0, 2})
    axes[1].set_title("2. Add a fill edge", pad=10)
    axes[1].text(0, -1.28, "Dashed edge: (1,3)", ha="center", fontsize=11, color=MUTED)
    ax = axes[2]
    ax.set_title("3. Two smaller cones", pad=10)
    ax.axis("off")
    for y, label, color in ((0.70, "PSD(3): {1,2,3}", BLUE), (0.23, "PSD(3): {1,3,4}", TEAL)):
        ax.add_patch(
            Rectangle(
                (0.05, y - 0.12), 0.9, 0.24, facecolor=color, alpha=0.10, edgecolor=color, lw=2
            )
        )
        ax.text(0.5, y, label, ha="center", va="center", fontsize=14, weight="bold")
    ax.annotate(
        "",
        xy=(0.5, 0.36),
        xytext=(0.5, 0.56),
        arrowprops={"arrowstyle": "<->", "color": AMBER, "lw": 2},
    )
    ax.text(
        0.5,
        0.45,
        "separator {1,3}",
        ha="center",
        va="center",
        color=AMBER,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 3},
    )
    ax.text(0.5, 0.01, "Original fixed zeros stay fixed", ha="center", color=MUTED, fontsize=11)
    save(fig, "chordal-cycle.svg")


def exactness_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
    ax = axes[0]
    ax.add_patch(Rectangle((-1, -1), 2, 2, facecolor="#fff3df", edgecolor=AMBER, lw=2))
    ax.add_patch(Circle((0, 0), 1, facecolor="#dbeafe", edgecolor=BLUE, lw=2))
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.axvline(0, color=MUTED, lw=0.5)
    ax.scatter([0.9], [0.9], c=RED, s=55, zorder=5)
    ax.annotate(
        "(0.9, 0.9)\nindefinite",
        (0.9, 0.9),
        xytext=(0.15, 1.13),
        color=RED,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    ax.text(0, -0.30, "$a^2+b^2 \\leq 1$", ha="center", color=BLUE, fontsize=16)
    ax.set(xlim=(-1.3, 1.3), ylim=(-1.25, 1.65), xlabel="$a$", ylabel="$b$", aspect="equal")
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_title("Fixed-zero PSD: the blue disk", fontsize=13)
    graph(axes[1], {0: (-0.9, 0), 1: (0, 0), 2: (0.9, 0)}, [(0, 1), (1, 2)], separator={1})
    axes[1].set_title("One shared diagonal, two contributions", fontsize=13)
    axes[1].text(-0.65, 0.45, "$u \\geq a^2$", ha="center", color=BLUE, fontsize=16)
    axes[1].text(0.65, 0.45, "$1-u \\geq b^2$", ha="center", color=TEAL, fontsize=16)
    axes[1].text(0, -0.55, "$u + (1-u) = 1$", ha="center", fontsize=17)
    axes[1].text(
        0,
        -1.0,
        "The orange square only checks\nthe original clique submatrices.",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    save(fig, "chordal-exactness.svg")


def pattern(ax, matrix, title, subtitle):
    ax.imshow(
        matrix,
        cmap=ListedColormap(["#ffffff", BLUE, "#fbbf24"]),
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel(subtitle, fontsize=11, labelpad=7)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cbd5e1")


def patterns_figure():
    d = 24
    block = np.zeros((d, d), dtype=int)
    for start in range(0, d, 6):
        block[start : start + 6, start : start + 6] = 1
    path = (np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]) <= 1).astype(int)
    cycle = path.copy()
    cycle[0, -1] = cycle[-1, 0] = 1
    bipartite = np.eye(16, dtype=int)
    bipartite[:8, 8:] = bipartite[8:, :8] = 1
    grid = np.eye(25, dtype=int)
    for row in range(5):
        for col in range(5):
            i = row * 5 + col
            neighbors = ((i + 1,) if col < 4 else ()) + ((i + 5,) if row < 4 else ())
            for j in neighbors:
                grid[i, j] = grid[j, i] = 1
    panels = [
        (block, "Block diagonal", "Disjoint cliques, no overlaps"),
        (path, "Narrow band", "Small cliques and separators"),
        (cycle, "Cycle", "Sparse; first add a chordal extension"),
        (bipartite, "Complete bipartite", "Large overlap can erase the benefit"),
        (np.ones((24, 24)), "Dense aggregate", "No split, even with low-rank data"),
        (grid, "Two-dimensional grid", "Fill and growing separators"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), layout="constrained")
    for ax, (matrix, title, subtitle) in zip(axes.flat, panels):
        pattern(ax, matrix, title, subtitle)
    save(fig, "chordal-patterns.svg")


def bipartite_figure():
    k = 8
    original = np.eye(2 * k, dtype=int)
    original[:k, k:] = original[k:, :k] = 1
    filled = original.copy()
    filled[:k, :k] = np.where(original[:k, :k], 1, 2)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    pattern(axes[0], original, "Original PSD(16)", "Blue: aggregate support")
    pattern(axes[1], filled, "One valid extension", "Gold: fill in the first group")
    ax = axes[2]
    ax.set_title("Separate cliques can cost more", fontsize=13)
    categories = np.arange(2)
    before = [136, 4096]
    after = [360, 5832]
    ratios = np.array(after) / before
    ax.bar(categories - 0.19, [1, 1], 0.34, color=BLUE, label="Original")
    ax.bar(categories + 0.19, ratios, 0.34, color=AMBER, label="Eight PSD(9) cones")
    for x, b, a, ratio in zip(categories, before, after, ratios):
        ax.text(x - 0.19, 1.04, f"{b:,}", ha="center", fontsize=11)
        ax.text(x + 0.19, ratio + 0.04, f"{a:,}", ha="center", fontsize=11)
    ax.set_xticks(categories, ["PSD scalar\nrows", "Cubic cone\nwork proxy"])
    ax.set_ylabel("Relative to the original")
    ax.set_ylim(0, 3.25)
    ax.legend(loc="upper left", fontsize=10, frameon=False)
    save(fig, "chordal-bipartite.svg")


if __name__ == "__main__":
    cycle_figure()
    exactness_figure()
    patterns_figure()
    bipartite_figure()
    print(f"Wrote four SVG figures to {OUTPUT}")
