"""Render the evaluation ladder used at the top of the README.

    python docs/figures/make_hero.py

Writes hero_ladder.png and hero_ladder-dark.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

THEMES = {
    "light": dict(bg="white", ink="#1c2530", muted="#5b6875", line="#b9c3cf",
                  rungs=["#eef2f6", "#e7eef6", "#f4ece5", "#fbeee7"],
                  edges=["#b9c3cf", "#7fa3c6", "#d69a72", "#c8683f"], arrow="#8a94a0"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", line="#3d444d",
                 rungs=["#161b22", "#12202f", "#241a12", "#2a1c14"],
                 edges=["#3d444d", "#4a7fb5", "#a05f38", "#e08a5c"], arrow="#6e7681"),
}

RUNGS = [
    ("Full-data bake-off", "same user, same position", "split by trial"),
    ("Feature sets", "same split and model", "Hudgins-4 against rich-14"),
    ("Paper protocol", "held-out arm position", "train at one, test at another"),
    ("Leave-one-subject-out", "held-out participant", "eight folds"),
]

HERE = os.path.dirname(os.path.abspath(__file__))


def render(theme, out):
    T = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=170)
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 46)
    ax.axis("off")
    fig.patch.set_facecolor(T["bg"])

    W, H, X0 = 76.0, 8.6, 15.0
    top = 36.0
    for i, (title, held, how) in enumerate(RUNGS):
        y = top - i * 10.2
        ax.add_patch(FancyBboxPatch((X0, y), W, H, boxstyle="round,pad=0,rounding_size=1.3",
                                    linewidth=1.4, edgecolor=T["edges"][i],
                                    facecolor=T["rungs"][i], zorder=2))
        ax.text(X0 + 3.0, y + H / 2 + 1.6, title, ha="left", va="center",
                fontsize=11.4, color=T["ink"], fontweight="bold", zorder=3)
        ax.text(X0 + 3.0, y + H / 2 - 2.2, f"{held}   ·   {how}", ha="left", va="center",
                fontsize=9.0, color=T["muted"], zorder=3)
        if i < len(RUNGS) - 1:
            ax.add_patch(FancyArrowPatch((X0 + W / 2, y), (X0 + W / 2, y - 1.6),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.3, color=T["arrow"], zorder=1))

    # difficulty axis on the left
    ax.add_patch(FancyArrowPatch((9.0, top + H - 1.0), (9.0, top - 3 * 10.2 + 1.0),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.6, color=T["arrow"], zorder=1))
    ax.text(6.4, (top + H + top - 3 * 10.2) / 2, "harder", rotation=90,
            ha="center", va="center", fontsize=10.5, color=T["muted"], fontweight="bold")

    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    render("light", os.path.join(HERE, "hero_ladder.png"))
    render("dark", os.path.join(HERE, "hero_ladder-dark.png"))
