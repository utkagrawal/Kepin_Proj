#!/usr/bin/env python3
"""Figures for the FD002 report: per-arm test RMSE (individual runs, mean ± std, ensemble)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_DIR, "..", "runs")
OUT = os.path.join(_DIR, "fd002")

ARMS = [  # (folder, label) top to bottom
    ("svd_static", "Original KePIN"),
    ("svd_static_nokooploss", "Original KePIN, Koopman switched off"),
    ("eig_static", "New fast layer, fixed rule"),
    ("eig_exo", "A: rule follows operating condition"),
    ("eig_health", "B: rule follows learned health"),
    ("eig_both", "A + B"),
    ("eig_static_ttf", "Koopman time-to-failure"),
    ("eig_static_wear", "Koopman time-to-failure + wear"),
]
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"
MEAN_C, ENS_C = "#2a78d6", "#eb6834"      # categorical slots 1, 2 (validated pair)


def load(protocol, arm):
    files = sorted(glob.glob(os.path.join(RUNS, protocol, "FD002", arm, "run*_result.json")))
    rmse = np.array([json.load(open(f))["rmse"] for f in files])
    preds = [np.load(f) for f in sorted(glob.glob(os.path.join(RUNS, protocol, "FD002", arm, "run*_predictions.npz")))]
    ens = np.mean([p["y_pred"] for p in preds], axis=0)
    ens_rmse = float(np.sqrt(np.mean((preds[0]["y_true"] - ens) ** 2)))
    return rmse, ens_rmse


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    titles = {"clean": "Fair test (clean protocol): main result",
              "paper": "Paper's method (test set picks the model)"}
    y = np.arange(len(ARMS))[::-1]
    for ax, protocol in zip(axes, ("clean", "paper")):
        for yi, (arm, _) in zip(y, ARMS):
            rmse, ens = load(protocol, arm)
            ax.scatter(rmse, np.full(len(rmse), yi + 0.18), s=22, color=MUTED, zorder=2,
                       label="single training run" if yi == y[0] else None)
            ax.errorbar(rmse.mean(), yi + 0.18, xerr=rmse.std(ddof=1), fmt="o", ms=8, color=MEAN_C,
                        elinewidth=2, capsize=0, zorder=3,
                        label="average of 3 runs (± std)" if yi == y[0] else None)
            ax.scatter(ens, yi - 0.18, marker="D", s=50, color=ENS_C, edgecolor="white", linewidth=1.5,
                       zorder=3, label="ensemble of 3 runs" if yi == y[0] else None)
            ax.text(ens - 0.06, yi - 0.18, f"{ens:.2f}", ha="right", va="center", fontsize=8, color=INK)
        if protocol == "paper":
            ax.axvline(15.08, color=INK, lw=1, ls="--", zorder=1)
            ax.text(15.08, y[0] + 0.62, " paper's KePIN: 15.08 (ensemble)", fontsize=8, color=INK, va="center")
        ax.set_title(titles[protocol], fontsize=11, color=INK, loc="left")
        ax.set_xlabel("test RMSE on FD002 (flights, lower is better)")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(14.2, 17.2)
    axes[0].set_yticks(y, [label for _, label in ARMS])
    axes[0].set_ylim(-0.7, len(ARMS) - 0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.6, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fd002_all_arms.png"), dpi=160)
    print("saved", os.path.join(OUT, "fd002_all_arms.png"))


if __name__ == "__main__":
    main()
