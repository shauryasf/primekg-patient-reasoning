#!/usr/bin/env python
"""Draw the explanatory diagrams shared by the team guide and the panel deck."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
INK, MUTED, LINE = "#12263A", "#5B6B7B", "#D8E0E8"
NAVY, BLUE, RED, GREEN, ORANGE, VIOLET = "#12355B", "#1C5CAB", "#C93534", "#12875C", "#C9501F", "#6250D6"
TINT, RED_TINT, GREEN_TINT, ORANGE_TINT, VIOLET_TINT = "#E6EEF7", "#F9E3E2", "#DFF2EA", "#FBE7DD", "#E8E5FA"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fill=TINT, edge=None, size=10.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=fill, edgecolor=edge or fill, linewidth=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=color, weight=weight, linespacing=1.25)


def arrow(ax, a, b, color=MUTED, style="-|>", width=1.4, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, linewidth=width, linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def censoring():
    rows = [("Patient A", 14, True), ("Patient B", 30, False), ("Patient C", 55, True), ("Patient D", 72, False), ("Patient E", 21, True), ("Patient F", 96, False)]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    for index, (name, month, progressed) in enumerate(rows):
        y = len(rows) - index
        ax.plot([0, month], [y, y], color=BLUE, linewidth=2.2, solid_capstyle="round")
        if progressed:
            ax.scatter([month], [y], marker="X", s=95, color=RED, zorder=3, edgecolor="white", linewidth=0.8)
        else:
            ax.scatter([month], [y], marker="o", s=62, facecolor="white", edgecolor=BLUE, linewidth=1.8, zorder=3)
        ax.text(-2, y, name, ha="right", va="center", fontsize=10.5)
    ax.axvline(36, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(36.8, 6.72, "36-month landmark", fontsize=9.5, color=MUTED)
    ax.scatter([], [], marker="X", s=80, color=RED, label="Progression recorded (event)")
    ax.scatter([], [], marker="o", s=55, facecolor="white", edgecolor=BLUE, linewidth=1.8, label="Follow-up ended, no event yet (censored)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=10)
    ax.set_xlim(0, 104)
    ax.set_ylim(0.4, 7.1)
    ax.set_yticks([])
    ax.set_xlabel("Months since diagnosis")
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "fig_censoring.png")


def cross_validation():
    fig, ax = canvas(8.4, 3.0, (0, 10.4), (-0.3, 5.7))
    for row in range(5):
        y = 4.6 - row
        ax.text(0.0, y + 0.3, f"Round {row + 1}", va="center", fontsize=10)
        for cell in range(5):
            held = cell == row
            box(ax, 1.15 + cell * 1.35, y, 1.25, 0.62, "held out" if held else "train", fill=RED if held else TINT, size=9.5, color="white" if held else INK, weight="bold" if held else "normal")
    ax.text(8.1, 3.6, "Every patient is\npredicted once, by\nmodels that never\nsaw that patient.\nThis is an\nout-of-fold result.", va="center", fontsize=10.5, linespacing=1.35)
    save(fig, "fig_cross_validation.png")


def hops():
    fig, ax = canvas(8.6, 4.3, (-5.0, 5.7), (-2.7, 2.65))
    ax.set_aspect("equal")
    for radius, color in ((4.7, "#F1F4F8"), (3.2, "#E6EEF7"), (1.75, "#CFE0F1")):
        ax.add_patch(matplotlib.patches.Ellipse((-1.1, 0), radius * 1.75, radius * 1.02, facecolor=color, edgecolor="white", linewidth=2))
    def node(x, y, text, color, r=0.36):
        ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=1.2, zorder=4))
        ax.text(x, y, text, ha="center", va="center", color="white", fontsize=10, weight="bold", zorder=5)
    ax.plot([-1.9, -1.1], [0, 0], color=MUTED, zorder=3)
    node(-1.9, 0, "Pt", NAVY)
    node(-1.1, 0, "Gene", BLUE, 0.4)
    ax.plot([-1.1, 0.0], [0, 0.48], color=MUTED, zorder=3); node(0.0, 0.48, "Path", GREEN)
    ax.plot([-1.1, -0.2], [0, -0.55], color=MUTED, zorder=3); node(-0.2, -0.55, "Drug", ORANGE)
    ax.plot([0.0, 1.15], [0.48, 0.72], color=MUTED, zorder=3); node(1.15, 0.72, "Gene", "#7FA6D3")
    ax.plot([1.15, 2.2], [0.72, 0.55], color=MUTED, linestyle="--", zorder=3); node(2.2, 0.55, "Path", "#8CC7AF")
    ax.text(-1.1, 1.0, "Hop 1", ha="center", fontsize=11, color=NAVY, weight="bold")
    ax.text(-1.1, 1.42, "Hop 2", ha="center", fontsize=11, color=NAVY, weight="bold")
    ax.text(-1.1, 2.03, "Hop 3", ha="center", fontsize=11, color=MUTED, weight="bold")
    ax.text(3.75, 1.5, "Fixed graph:\nhops 0, 1, 2\n(final system)", fontsize=12.5, color=NAVY, weight="bold", va="center")
    ax.text(3.75, -1.5, "GSL also tried\nhop 3, and learned\nedge weights", fontsize=12.5, color=RED, weight="bold", va="center")
    ax.text(-1.1, -2.62, "Hop 0: patient, sample, altered genes", ha="center", fontsize=11, color=MUTED)
    save(fig, "fig_hops.png")


def message_passing():
    fig, ax = canvas(9.0, 2.5, (0, 12), (0, 3.2))
    positions = {"P": (1.1, 1.6), "S": (2.6, 1.6), "G1": (4.2, 2.45), "G2": (4.2, 0.75), "W": (5.9, 2.45)}
    labels = {"P": ("Patient", NAVY), "S": ("Sample", MUTED), "G1": ("TP53", BLUE), "G2": ("MYC", BLUE), "W": ("Pathway", GREEN)}
    for a, b in (("P", "S"), ("S", "G1"), ("S", "G2"), ("G1", "W")):
        arrow(ax, positions[b], positions[a], color=BLUE, width=1.6)
    for key, (x, y) in positions.items():
        name, color = labels[key]
        box(ax, x - 0.58, y - 0.3, 1.16, 0.6, name, fill=color, size=9.5, color="white", weight="bold")
    ax.text(7.2, 1.6, "Each layer, every node collects its\nneighbours' features and updates itself.\nThree layers carry pathway context\nback to the patient node, which then\nproduces the risk score.", va="center", fontsize=10.5, linespacing=1.4)
    save(fig, "fig_message_passing.png")


def rl_loop():
    fig, ax = canvas(9.4, 3.55, (0, 14), (0, 5))
    box(ax, 0.2, 3.1, 2.5, 1.1, "Neural network\nwrites a rule,\none fact at a time", fill=NAVY, color="white", size=10)
    box(ax, 3.6, 3.1, 2.5, 1.1, "Candidate rule\nA and B and C", fill=TINT)
    box(ax, 7.0, 3.1, 3.0, 1.1, "Reward\nonly if it works on two\nseparate patient groups", fill=ORANGE_TINT, size=10)
    arrow(ax, (2.75, 3.65), (3.55, 3.65))
    arrow(ax, (6.15, 3.65), (6.95, 3.65))
    arrow(ax, (8.5, 3.05), (1.45, 3.05), color=RED, rad=-0.28)
    ax.text(5.0, 1.72, "Rules that earn reward become more likely", ha="center", fontsize=10, color=RED)
    box(ax, 11.0, 3.1, 2.8, 1.1, "Final check\nmust hold on patients\nit never saw", fill=GREEN_TINT, size=10)
    arrow(ax, (10.05, 3.65), (10.95, 3.65))
    box(ax, 11.0, 0.5, 2.8, 1.25, "Markov logic\nlearns every weight,\ndrops the rest", fill=NAVY, color="white", size=10)
    arrow(ax, (12.4, 3.05), (12.4, 1.8))
    ax.text(12.55, 2.42, "best rules", fontsize=9.5, color=MUTED)
    save(fig, "fig_rl_loop.png")


def timeline():
    steps = [
        ("Milestone 1", "Patient graphs", "Binary PFS label:\nAUROC 0.54"),
        ("Milestone 2", "36-month label\nand PrimeKG", "Clinical 0.70\nGCN 0.68"),
        ("Milestone 3", "7 GNNs, C-index\nand Cox losses", "No model beats\nclinical"),
        ("Milestone 4", "Graph structure\nlearning", "No real gain,\nso dropped"),
        ("Milestone 5", "Neuro-symbolic\nreports (RuleKit)", "Tied at 0.67,\nnow explained"),
        ("Milestone 6", "MLN, PSL, RL\nlogic reasoners", "69% of formulas\nuse PrimeKG"),
    ]
    fig, ax = canvas(11.2, 2.9, (0, 12), (0, 3.6))
    ax.plot([0.5, 11.5], [2.05, 2.05], color=LINE, linewidth=3, zorder=1)
    for index, (head, what, result) in enumerate(steps):
        x = 1.0 + index * 2.0
        final = index >= 4
        ax.scatter([x], [2.05], s=150, color=NAVY if final else BLUE, edgecolor="white", linewidth=2, zorder=3)
        ax.text(x, 3.3, head, ha="center", fontsize=9.5, color=MUTED, weight="bold")
        ax.text(x, 2.72, what, ha="center", va="center", fontsize=10.5, linespacing=1.2, weight="bold")
        ax.text(x, 1.3, result, ha="center", va="center", fontsize=10, color=INK, linespacing=1.25)
    save(fig, "fig_timeline.png")


def fusion():
    fig, ax = canvas(9.6, 3.1, (0, 13), (0, 4.3))
    box(ax, 0.2, 3.05, 3.6, 0.9, "Clinical block (19 fields)\nage, sex, subtype, stages", fill=TINT, size=10)
    box(ax, 0.2, 1.75, 3.6, 0.9, "GNN risk\nGraphSAGE", fill=TINT, size=10)
    box(ax, 0.2, 0.45, 3.6, 0.9, "Logic risk\nRL + Markov logic", fill=TINT, size=10)
    box(ax, 5.2, 1.5, 2.8, 1.4, "One Cox model\ncombines all\nthree inputs", fill=NAVY, color="white")
    for y in (3.5, 2.2, 0.9):
        arrow(ax, (3.85, y), (5.15, 2.2))
    box(ax, 9.4, 3.05, 3.4, 0.9, "Risk and survival curve", fill=GREEN_TINT, size=10)
    box(ax, 9.4, 1.75, 3.4, 0.9, "Month estimate", fill=ORANGE_TINT, size=10)
    box(ax, 9.4, 0.45, 3.4, 0.9, "Paths: 0.6 neural + 0.4 rule", fill=VIOLET_TINT, size=10)
    for y in (3.5, 2.2, 0.9):
        arrow(ax, (8.05, 2.2), (9.35, y))
    save(fig, "fig_fusion.png")


def saturation():
    rows = [("Clinical Cox (19 fields)", 0.675, 0.035, "c"), ("Neuro-symbolic Cox (chosen)", 0.664, 0.021, "n"), ("GNN + logic, no clinical", 0.642, 0.056, "o"), ("Survival MLN", 0.630, 0.055, "l"), ("RL-structured MLN (engine)", 0.628, 0.055, "l"), ("GraphSAGE, C-index loss", 0.622, 0.067, "g"), ("RuleKit rules", 0.619, 0.048, "l"), ("GCN, Cox loss", 0.612, 0.044, "g"), ("Survival PSL", 0.604, 0.059, "l"), ("R-GCN, Cox loss", 0.586, 0.041, "g"), ("RL rules alone", 0.563, 0.046, "l"), ("Random survival forest", 0.554, 0.023, "o")]
    colors = {"c": MUTED, "n": NAVY, "o": MUTED, "l": GREEN, "g": BLUE}
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for index, (name, mean, sd, kind) in enumerate(rows[::-1]):
        ax.errorbar(mean, index, xerr=sd, fmt="o", color=colors[kind], ecolor=LINE if kind != "n" else NAVY, elinewidth=2.4, markersize=7, capsize=0)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows[::-1]], color=INK, fontsize=10)
    ax.axvline(0.659, color=RED, linestyle="--", linewidth=1.2)
    ax.text(0.662, 11.42, "tie bar 0.659", color=RED, fontsize=9.5)
    ax.set_xlabel("Harrell C-index: five-fold mean, line shows one standard deviation")
    ax.set_xlim(0.48, 0.73)
    ax.grid(axis="x", color="#EEF1F4")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for kind, label in (("g", "GNN"), ("l", "Logic or rules"), ("c", "Other baselines"), ("n", "Chosen model")):
        ax.scatter([], [], color=colors[kind], label=label, s=45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.14), ncol=4, frameon=False, fontsize=10)
    save(fig, "fig_saturation.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for draw in (censoring, cross_validation, hops, message_passing, rl_loop, timeline, fusion, saturation):
        draw()
    print(sorted(path.name for path in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
