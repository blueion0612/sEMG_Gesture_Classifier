"""Draw the README hero: the evaluation ladder.

    python docs/figures/make_hero.py

Writes hero_ladder.png and hero_ladder-dark.png. The repository commits no
numbers, so the hero is the structure of the evaluation rather than a result.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
sys.path.insert(0, HERE)

import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RUNGS = [
    ("Full-data bake-off", "same user, same position", "split by trial"),
    ("Feature sets", "same split and model", "Hudgins-4 against rich-14"),
    ("Paper protocol", "held-out arm position", "train at one, test at another"),
    ("Leave-one-subject-out", "held-out participant", "eight folds"),
]


def ladder(T):
    fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 4.6))
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 46)
    ax.axis("off")
    # neutral, green, gold, gold with the ink edge: the rungs get harder downwards
    edges = [T["line"], T["green"], T["gold"], T["ink"]]
    fills = [T["fill"], T["green_fill"], T["gold_fill"], T["gold_fill"]]

    W, H, X0 = 76.0, 8.6, 15.0
    top = 36.0
    for i, (title, held, how) in enumerate(RUNGS):
        y = top - i * 10.2
        ax.add_patch(FancyBboxPatch((X0, y), W, H, boxstyle="round,pad=0,rounding_size=1.3",
                                    linewidth=1.4, edgecolor=edges[i], facecolor=fills[i], zorder=2))
        ax.text(X0 + 3.0, y + H / 2 + 1.6, title, ha="left", va="center",
                fontsize=figstyle.TITLE, color=T["ink"], fontweight="bold", zorder=3)
        ax.text(X0 + 3.0, y + H / 2 - 2.2, f"{held}   ·   {how}", ha="left", va="center",
                fontsize=figstyle.SMALL, color=T["muted"], zorder=3)
        if i < len(RUNGS) - 1:
            ax.add_patch(FancyArrowPatch((X0 + W / 2, y), (X0 + W / 2, y - 1.6),
                                         arrowstyle="-|>", mutation_scale=11,
                                         linewidth=1.3, color=T["muted"], zorder=1))

    # difficulty axis on the left
    ax.add_patch(FancyArrowPatch((9.0, top + H - 1.0), (9.0, top - 3 * 10.2 + 1.0),
                                 arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.6, color=T["muted"], zorder=1))
    ax.text(6.4, (top + H + top - 3 * 10.2) / 2, "harder", rotation=90,
            ha="center", va="center", fontsize=figstyle.BODY, color=T["muted"], fontweight="bold")
    return fig


if __name__ == "__main__":
    figstyle.save_both(ladder, HERE + "hero_ladder")
