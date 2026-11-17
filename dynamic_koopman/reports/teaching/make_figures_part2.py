#!/usr/bin/env python3
"""Figures for Part 2 of KePIN_Project_Explained.pdf (adaptive Koopman, 22 Oct - 9 Nov 2026)."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, OUT, RED, VIOLET, arrow, box, canvas, save)

DK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


# ---------------------------------------------------------------- 13. the adaptive-K idea
def adaptive_idea():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    k = np.arange(0, 31)
    for a, (title, rhos) in zip(ax, [("Static K: every engine forgets at the same speed", (0.93, 0.93)),
                                    ("Adaptive K (the idea): the speed depends on health", (0.97, 0.80))]):
        a.plot(k, rhos[0] ** k, color=BLUE, lw=2, label="healthy engine")
        a.plot(k, rhos[1] ** k, color=ORANGE, lw=2, ls="--" if rhos[0] == rhos[1] else "-",
               label="engine near failure")
        a.set(xlabel="steps ahead (k)", ylabel="size of a mode  (ρᵏ)", ylim=(0, 1.05), title=title)
        a.title.set_fontsize(10)
        a.grid(color=GRID)
        a.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    save(fig, "13_adaptive_idea.png")


# ---------------------------------------------------------------- 16. curvature through the encoder
def curvature_layers():
    d = json.load(open(os.path.join(DK, "reports", "latent_curvature", "layers.json")))
    short = ["input", "proj+BN\n+ReLU", "conv 0", "conv 1", "conv 2", "conv 3", "BiLSTM", "LN", "proj+BN",
             "attention", "feed-fwd", "latent\nproj", "latent z"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for (name, v), c in zip(d.items(), [GREEN, ORANGE]):
        y = [x["curvature"] for x in v.values()]
        lab = "self-supervised encoder (no RUL label)" if "self" in name else "RUL-trained encoder"
        ax.plot(range(len(y)), y, "o-", color=c, lw=2, ms=5, label=lab)
    ax.axhline(1, color=MUTED, lw=1, ls=":")
    ax.text(12.2, 1.08, "1× = straight line", color=MUTED, fontsize=8, ha="right")
    ax.annotate("first ReLU: biggest drop", xy=(1, 3.4), xytext=(2.3, 4.6), fontsize=9,
                arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xticks(range(len(short)), short, fontsize=8)
    ax.set(ylabel="speed-up near failure\n(late slope ÷ mid-life slope)", ylim=(0, 5.6))
    ax.grid(axis="y", color=GRID)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("How much of the 'speeding up near failure' survives each encoder layer (FD002)", loc="left", fontsize=11)
    plt.tight_layout()
    save(fig, "16_curvature_layers.png")


# ---------------------------------------------------------------- 17. hypernetwork architecture
def hyper_architecture():
    fig, ax = canvas(12.4, 6.8, 12.4, 7.4)
    # main KePIN path (top)
    box(ax, 0.1, 5.9, 1.8, 1.0, "Last 30 flights\n(one window)", "blue")
    box(ax, 2.2, 5.9, 2.8, 1.0, "KePIN ENCODER\nConv → BiLSTM → Attention", "blue", bold=True)
    box(ax, 5.3, 5.9, 1.8, 1.0, "hidden state z\n(30 × 128)", "blue")
    box(ax, 7.4, 5.9, 1.8, 1.0, "average & max\nover time", "blue")
    box(ax, 10.0, 4.2, 1.9, 1.2, "PREDICTOR\n(dense layers)", "green", bold=True)
    box(ax, 10.0, 2.9, 1.9, 0.8, "RUL", "green")
    arrow(ax, 1.9, 6.4, 2.2, 6.4); arrow(ax, 5.0, 6.4, 5.3, 6.4); arrow(ax, 7.1, 6.4, 7.4, 6.4)
    arrow(ax, 9.2, 6.2, 10.6, 5.4); arrow(ax, 10.95, 4.2, 10.95, 3.7, GREEN)
    # history -> health
    box(ax, 0.1, 3.7, 1.8, 1.4, "Engine history\nflights 1…t\n42 numbers\n(drift + rate)", "violet")
    box(ax, 2.2, 3.85, 1.9, 1.1, "health net\nDense 32 → 1\nsigmoid", "violet", bold=True)
    box(ax, 4.4, 3.85, 1.2, 1.1, "health\nh ∈ (0,1)", "violet")
    arrow(ax, 1.9, 4.4, 2.2, 4.4, VIOLET); arrow(ax, 4.1, 4.4, 4.4, 4.4, VIOLET)
    box(ax, 4.4, 2.75, 1.2, 0.75, "age t/100\n(arm B)", "grey", ls="--")
    # hypernetwork
    box(ax, 5.95, 3.1, 2.0, 1.85, "HYPERNETWORK\nDense 64 (ReLU)\n→ Dense\n(starts at 0)", "orange", bold=True)
    arrow(ax, 5.6, 4.4, 5.95, 4.2, VIOLET); arrow(ax, 5.6, 3.1, 5.95, 3.5, MUTED, "--")
    box(ax, 5.95, 1.3, 2.0, 1.3, "per engine:\nΔs_k (decay)\nΔθ_k (rotation)\ng_k on/off (arm B)", "orange")
    arrow(ax, 6.95, 3.1, 6.95, 2.6, ORANGE)
    # operator
    box(ax, 8.3, 1.3, 3.6, 1.3, "K_engine = P · diag(ρ_k R(θ_k)) · Pᵀ\nρ_k = g_k · sigmoid(s_k + Δs_k)\nP, s_k, θ_k shared by all engines", "orange")
    arrow(ax, 7.95, 1.95, 8.3, 1.95, ORANGE)
    box(ax, 8.3, 0.1, 3.6, 0.85, "eigenvalues → 12 spectral features\n(now different for every engine)", "red")
    arrow(ax, 10.1, 1.3, 10.1, 0.95, ORANGE)
    ax.add_patch(matplotlib.patches.FancyArrowPatch((11.9, 0.55), (11.75, 4.2), connectionstyle="arc3,rad=0.25",
                 arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.6))
    # training-only parts
    box(ax, 0.1, 1.3, 4.0, 1.3, "Koopman training losses (as in KePIN):\nK_engine must move z one and several\nsteps ahead, so the generated K stays\na real rule of motion", "orange", ls="--")
    box(ax, 0.1, 0.1, 4.0, 0.85, "mono loss (no RUL labels):\nh must fall over each engine's life", "violet", ls="--")
    ax.text(4.3, 0.75, "dashed = used only\nduring training", fontsize=8.5, color=MUTED, va="center")
    ax.text(0.1, 7.25, "Purple = health regime    Orange = Koopman    Red = the only road from K to the RUL",
            fontsize=9, color=MUTED, va="center")
    save(fig, "17_hyper_architecture.png")


# ---------------------------------------------------------------- 18. results
def hyper_results():
    arms = ["Original KePIN\n(static K)", "Hyper: window\ncontext", "Hyper: age\n+ on/off modes",
            "A: K from\nhealth h only", "B: K from h + age\n+ on/off modes", "C (control): static K,\nh + age to predictor"]
    paper = [14.91, 14.72, 14.04, 13.09, 11.03, 11.04]
    clean = [15.34, np.nan, np.nan, 13.70, 11.77, 11.99]
    y = np.arange(len(arms))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(y + 0.19, paper, 0.36, color=BLUE, label="paper's method (test set picks the epoch)")
    ax.barh(y - 0.19, clean, 0.36, color=ORANGE, label="fair test (held-out training engines pick it)")
    for yy, p, c in zip(y, paper, clean):
        ax.text(p + 0.1, yy + 0.19, f"{p:.2f}", va="center", fontsize=8.5, color=INK)
        if not np.isnan(c):
            ax.text(c + 0.1, yy - 0.19, f"{c:.2f}", va="center", fontsize=8.5, color=INK)
        else:
            ax.text(10.1, yy - 0.19, "not run", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, arms, fontsize=8.5)
    ax.set_xlim(10, 16.3)
    ax.set_xlabel("ensemble RMSE (3 seeds, FD002) — lower is better")
    ax.grid(axis="x", color=GRID)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.axhline(2.5, color=MUTED, lw=0.8, ls=":")
    ax.text(16.25, 2.62, "with engine-history health ↓", fontsize=8, color=MUTED, ha="right")
    plt.tight_layout()
    save(fig, "18_hyper_results.png")


if __name__ == "__main__":
    adaptive_idea()
    curvature_layers()
    hyper_architecture()
    hyper_results()
    print("ok")
