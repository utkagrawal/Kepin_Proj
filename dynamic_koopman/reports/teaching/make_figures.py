#!/usr/bin/env python3
"""Diagrams and explanatory figures for the teaching PDF (KePIN_Project_Explained.pdf)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kepin_core")
INK, MUTED, GRID = "#1d1d1b", "#7a7974", "#e4e3df"
BLUE, ORANGE, GREEN, YELLOW, PINK, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                  "#e87ba4", "#6250d6", "#e34948")
LIGHT = {"blue": "#dbe9fa", "orange": "#fde3d7", "green": "#d5f2e6", "grey": "#efeeea",
         "red": "#fadcdc", "violet": "#e4e0fa"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc="grey", ec=INK, fs=9, bold=False, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=LIGHT.get(fc, fc), ec=ec, lw=1.4, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, wrap=True)


def arrow(ax, x1, y1, x2, y2, color=INK, ls="-", lw=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=lw, ls=ls))


def canvas(w, h, xmax=10, ymax=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. original KePIN
def kepin_architecture():
    fig, ax = canvas(10, 5.2, 10, 6.2)
    box(ax, 0.1, 4.6, 1.9, 1.1, "Last 30 flights\n× sensors\n(one window)", "blue")
    box(ax, 2.3, 4.6, 1.6, 1.1, "EMA\nsmoothing", "grey")
    box(ax, 4.2, 4.6, 3.1, 1.1, "ENCODER\nConv + SE blocks → BiLSTM\n→ Attention", "blue", bold=True)
    box(ax, 7.6, 4.6, 2.2, 1.1, "hidden state z\n(30 steps × 96–128\nnumbers)", "blue")
    arrow(ax, 2.0, 5.15, 2.3, 5.15); arrow(ax, 3.9, 5.15, 4.2, 5.15); arrow(ax, 7.3, 5.15, 7.6, 5.15)
    # pooled path
    box(ax, 7.6, 2.8, 2.2, 1.0, "average & max\nover time\n(192–256 numbers)", "blue")
    arrow(ax, 8.7, 4.6, 8.7, 3.8)
    # koopman path
    box(ax, 4.2, 2.8, 3.1, 1.0, "KOOPMAN rule K\n(one fixed matrix,\nsame for every engine)", "orange", bold=True)
    arrow(ax, 7.6, 5.0, 6.6, 3.8, ORANGE)
    box(ax, 4.2, 1.2, 3.1, 1.0, "12 'spectral' numbers\n(decay rates, frequencies…)\nIDENTICAL for all engines", "red")
    arrow(ax, 5.75, 2.8, 5.75, 2.2, ORANGE)
    box(ax, 0.3, 2.8, 3.4, 1.0, "Koopman training losses\n(predict z 1–3 steps ahead)\n– only during training –", "orange", ls="--")
    arrow(ax, 4.2, 3.3, 3.7, 3.3, ORANGE, "--")
    box(ax, 7.6, 1.2, 2.2, 1.0, "PREDICTOR\n(dense layers)", "green", bold=True)
    arrow(ax, 8.7, 2.8, 8.7, 2.2, GREEN)
    arrow(ax, 7.3, 1.7, 7.6, 1.7, RED)
    box(ax, 7.6, 0.05, 2.2, 0.75, "RUL (flights left)", "green")
    arrow(ax, 8.7, 1.2, 8.7, 0.8, GREEN)
    ax.text(0.1, 0.35, "Red box = the problem: the only way Koopman reaches the\nprediction is 12 numbers that never change between engines.",
            fontsize=9, color=RED, va="center")
    ax.set_title("Original KePIN: how one prediction is made", loc="left", fontsize=12, color=INK)
    save(fig, "01_kepin_architecture.png")


# ---------------------------------------------------------------- 3. ablation bars
def ablation():
    ds = ["FD001", "FD002", "FD003", "FD004"]
    w_ = [12.92, 15.00, 11.56, 17.11]
    wo = [13.60, 14.73, 11.47, 17.23]
    p = [0.21, 0.39, 0.80, 0.22]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(4)
    ax.bar(x - 0.19, w_, 0.36, color=BLUE, label="KePIN with Koopman")
    ax.bar(x + 0.19, wo, 0.36, color=ORANGE, label="KePIN with Koopman removed")
    for i in range(4):
        ax.text(x[i] - 0.19, w_[i] + 0.15, f"{w_[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.19, wo[i] + 0.15, f"{wo[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i], max(w_[i], wo[i]) + 0.9, f"p = {p[i]:.2f} (n.s.)", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, ds)
    ax.set_ylabel("RMSE (ensemble of 3), lower = better")
    ax.set_ylim(0, 21)
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2)
    ax.set_title("Removing Koopman completely: no significant change on any dataset", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "03_ablation.png")


# ---------------------------------------------------------------- 5. why dynamic failed
def dynamic_failure():
    fig, ax = canvas(10, 3.6, 10, 4.2)
    box(ax, 0.1, 2.6, 2.2, 1.1, "hidden state z\n(192–256 numbers:\nall the information)", "blue")
    box(ax, 3.0, 2.6, 2.6, 1.1, "dynamic Koopman rule\n(changes with condition\nor learned 'health')", "orange")
    box(ax, 6.2, 2.6, 1.9, 1.1, "12 numbers\n(a SUMMARY of z)", "red")
    box(ax, 8.4, 1.4, 1.5, 1.1, "PREDICTOR", "green", bold=True)
    arrow(ax, 2.3, 3.15, 3.0, 3.15, ORANGE); arrow(ax, 5.6, 3.15, 6.2, 3.15, ORANGE)
    arrow(ax, 8.1, 3.0, 8.6, 2.5, RED, "--")
    arrow(ax, 1.2, 2.6, 8.4, 1.8, BLUE, lw=2.6)
    ax.text(3.2, 2.25, "direct, easy path: the full information", fontsize=9, color=BLUE, rotation=-5)
    ax.text(0.1, 0.6, "The dynamic rule only re-packages information the predictor already receives in full.\n"
            "A summary of something you already know adds nothing, so training learns to ignore it\n"
            "(replacing the 12 numbers by a constant changed predictions by < 0.1 flights).",
            fontsize=9, color=INK)
    ax.set_title("Why making Koopman 'dynamic' could not help", loc="left", fontsize=12)
    save(fig, "05_dynamic_failure.png")


# ---------------------------------------------------------------- 6. fade vs grow
def fade_vs_grow():
    k = np.arange(0, 126)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for rho, c in ((0.99, BLUE), (0.95, GREEN), (0.80, ORANGE)):
        axes[0].plot(k, -3 - 3 * rho ** k, color=c, lw=2, label=f"mode fades by ×{rho} per flight")
    axes[0].axhline(0, color=INK, ls="--", lw=1)
    axes[0].axhline(-3, color=MUTED, ls=":", lw=1)
    axes[0].text(60, 0.25, "failure line", fontsize=8)
    axes[0].text(60, -2.75, "resting level the curves fade back to", fontsize=8, color=MUTED)
    axes[0].set_ylim(-6.5, 1)
    axes[0].set_title("A stable Koopman rule can only FADE\n(health drifts back to a level, never crosses)", loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[1].plot(k, -6 + 0.06 * k, color=RED, lw=2, label="wear builds up: −6 + 0.06 × flights")
    axes[1].plot(k, -6 + 0.06 * k + 0.8 * np.sin(k / 9) * 0.97 ** k, color=VIOLET, lw=1.5, ls="--",
                 label="+ fading Koopman wiggle")
    axes[1].axhline(0, color=INK, ls="--", lw=1)
    axes[1].set_ylim(-6.5, 2)
    axes[1].plot(100, 0, "o", color=RED, ms=8)
    axes[1].text(102, -0.9, "crosses at\n~100 flights", fontsize=8, color=RED)
    axes[1].set_title("Real wear GROWS until failure\n(needs a mode that never fades)", loc="left", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    for a in axes:
        a.set_xlabel("flights into the future")
        a.set_ylabel("projected health")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "06_fade_vs_grow.png")


# ---------------------------------------------------------------- 7. wear model diagram
def wear_architecture():
    fig, ax = canvas(10, 4.2, 10, 5)
    box(ax, 0.1, 3.5, 2.0, 1.0, "30 flights of\nsensor data", "blue")
    box(ax, 2.5, 3.5, 2.2, 1.0, "ENCODER\n(same as KePIN)", "blue", bold=True)
    box(ax, 5.1, 3.5, 1.9, 1.0, "hidden state\n(today)", "blue")
    arrow(ax, 2.1, 4.0, 2.5, 4.0); arrow(ax, 4.7, 4.0, 5.1, 4.0)
    box(ax, 7.4, 3.9, 2.5, 0.8, "CURRENT HEALTH H₀", "green", bold=True)
    box(ax, 7.4, 2.9, 2.5, 0.8, "WEAR RATE v ≥ 0\n(per flight)", "orange", bold=True)
    arrow(ax, 7.0, 4.1, 7.4, 4.3, GREEN); arrow(ax, 7.0, 3.8, 7.4, 3.3, ORANGE)
    box(ax, 1.0, 1.1, 6.0, 1.3, "KOOPMAN fast-forward, flight by flight (k = 1 … 125):\n"
        "health(k) = Koopman fading part(k)  +  v × k   ← wear accumulator\n(a mode with eigenvalue exactly 1: it never fades)",
        "violet")
    arrow(ax, 8.6, 2.9, 6.6, 2.4, VIOLET)
    box(ax, 7.4, 1.1, 2.5, 1.3, "RUL = first flight\nwhere health(k)\nreaches 0", "green", bold=True)
    arrow(ax, 7.0, 1.75, 7.4, 1.75, GREEN)
    ax.text(0.1, 0.35, "No black-box predictor: the Koopman projection IS the prediction, and every engine gets an explanation:\n"
            "\"health is H₀, wearing v per flight → hits the failure line in about −H₀ / v flights\".", fontsize=9)
    ax.set_title("The wear model: Koopman becomes the predictor", loc="left", fontsize=12)
    save(fig, "07_wear_architecture.png")


# ---------------------------------------------------------------- 9. condition-aware architecture
def condition_aware_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax in axes:
        ax.set_xlim(0, 5); ax.set_ylim(0, 8); ax.axis("off")
    a, b = axes
    a.set_title("ORIGINAL KePIN", fontsize=12, loc="left")
    steps = [("sensors scaled with ONE global\naverage/spread for all 6 conditions", "grey"),
             ("EMA smoothing\n(before the model)", "grey"),
             ("ENCODER (conv + LSTM + attention)", "blue"),
             ("hidden state z", "blue"),
             ("Koopman rule on z → 12 constant numbers", "orange"),
             ("PREDICTOR reads z", "green")]
    for i, (t, c) in enumerate(steps):
        y = 6.9 - i * 1.25
        box(a, 0.3, y, 4.4, 0.9, t, c, fs=9)
        if i:
            arrow(a, 2.5, y + 1.25, 2.5, y + 0.9)
    b.set_title("NEW: condition-aware KePIN", fontsize=12, loc="left")
    steps = [("raw sensors + flying condition of every flight", "grey"),
             ("[NEW 1] condition-aware input layer:\nown learned scale & offset PER CONDITION", "red"),
             ("[MOVED 2] EMA smoothing inside the model,\nAFTER step 1", "red"),
             ("ENCODER (unchanged)", "blue"),
             ("[NEW 3] Koopman with control:\nz = s (wear state) + C × condition", "orange"),
             ("Koopman rule + PREDICTOR read s", "green")]
    for i, (t, c) in enumerate(steps):
        y = 6.9 - i * 1.25
        box(b, 0.3, y, 4.4, 0.9, t, c, fs=9, bold=c == "red")
        if i:
            arrow(b, 2.5, y + 1.25, 2.5, y + 0.9)
    fig.text(0.02, 0.01, "Red = changes that produced the accuracy gain. Orange [NEW 3] turned out not to work as designed (see the "
             "'Koopman with control' section).", fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "09_condition_aware_architecture.png")


# ---------------------------------------------------------------- 10. why per-condition normalisation (REAL DATA)
def condition_norm_real_data():
    cfg = json.load(open(os.path.join(CORE, "datasets_kepin_config.json")))[1]
    df = pd.read_csv(os.path.join(CORE, cfg["train_path"]), sep=r"\s+", header=None)
    df = df.iloc[:, :len(cfg["column_names"])]
    df.columns = cfg["column_names"]
    key = df["setting1"].round(0).astype(str) + "_" + df["setting2"].round(2).astype(str) + "_" + df["setting3"].round(0).astype(str)
    df["cond"] = pd.factorize(key, sort=True)[0]
    sensor = "s11"
    g = (df[sensor] - df[sensor].mean()) / df[sensor].std()                 # KePIN: one global scaling
    stats = df.groupby("cond")[sensor].agg(["mean", "std"])
    c = (df[sensor] - stats.loc[df["cond"], "mean"].values) / stats.loc[df["cond"], "std"].values
    eng = df["id"] == 1
    cyc = df.loc[eng, "cycle"].values
    cols = plt.cm.tab10(np.arange(6))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True, gridspec_kw={"hspace": 0.55})
    for k in range(6):
        m = (df.loc[eng, "cond"] == k).values
        axes[0].scatter(cyc[m], g[eng].values[m], s=9, color=cols[k], label=f"condition {k + 1}")
        axes[1].scatter(cyc[m], c[eng].values[m], s=9, color=cols[k])
    axes[0].set_title(f"(a) KePIN: sensor {sensor} of engine 1 (FD002), one global scaling. Colour = flying condition.\n"
                      "The value jumps between 6 levels every flight; the slow wear trend is invisible.", loc="left", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=6, loc="upper left", bbox_to_anchor=(0, -0.04), markerscale=1.5)
    trend = pd.Series(c[eng].values).rolling(15, min_periods=1).mean()
    axes[1].plot(cyc, trend, color=INK, lw=2, label="15-flight average")
    axes[1].set_title("(b) NEW: the same sensor scaled PER CONDITION. The jumps disappear and the wear trend\n"
                      "(rising towards failure at the right) becomes clearly visible.", loc="left", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    # EMA before vs after
    a = 2 / 31
    def ema(x):
        y = np.empty_like(x); y[0] = x[0]
        for i in range(1, len(x)):
            y[i] = a * x[i] + (1 - a) * y[i - 1]
        return y
    gw, cw = g[eng].values, c[eng].values
    axes[2].plot(cyc, ema(gw), color=ORANGE, lw=2, label="KePIN: smooth first (conditions blended together)")
    axes[2].plot(cyc, ema(cw), color=BLUE, lw=2, label="NEW: condition fix first, then smooth")
    axes[2].set_title("(c) Order matters: smoothing BEFORE the condition fix blends different conditions into a\n"
                      "noisy mixture; smoothing AFTER keeps a clean wear signal.", loc="left", fontsize=9.5)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    axes[2].set_xlabel("engine age (flights)")
    for ax in axes:
        ax.set_ylabel("scaled value")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    save(fig, "10_condition_norm_real_data.png")


# ---------------------------------------------------------------- 11. koopman with control: diagram + probe
def koopman_control():
    fig, ax = canvas(10, 2.8, 10, 3.2)
    box(ax, 0.1, 2.0, 2.2, 0.9, "flying condition\nof each flight", "grey")
    box(ax, 0.1, 0.4, 2.2, 0.9, "hidden state z\n(from encoder)", "blue")
    box(ax, 3.0, 2.0, 3.0, 0.9, "C × flying condition\n('flight effect', learned)", "grey")
    box(ax, 3.0, 0.4, 3.0, 0.9, "s = z − C × condition\n(intended: pure WEAR state)", "orange", bold=True)
    box(ax, 6.8, 0.4, 3.1, 0.9, "Koopman rule: s(next) = K s\n+ PREDICTOR reads s", "green")
    arrow(ax, 2.3, 2.45, 3.0, 2.45); arrow(ax, 2.3, 0.85, 3.0, 0.85)
    arrow(ax, 4.5, 2.0, 4.5, 1.3, INK, "--"); arrow(ax, 6.0, 0.85, 6.8, 0.85, GREEN)
    ax.text(4.6, 1.55, "subtract", fontsize=8, color=MUTED)
    ax.set_title("Koopman with control: the idea", loc="left", fontsize=12)
    save(fig, "11a_koopman_control_idea.png")

    names = ["Original KePIN\nhidden state", "+ condition-aware\ninput", "+ Koopman with control:\n'wear state' s"]
    cond = [0.263, 0.004, 0.968]
    rul = [0.887, 0.893, 0.891]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(3)[::-1]
    axes[0].barh(y, cond, color=[MUTED, GREEN, RED], height=0.6)
    axes[1].barh(y, rul, color=BLUE, height=0.6)
    for i in range(3):
        axes[0].text(cond[i] + 0.02, y[i], f"{cond[i]:.2f}", va="center", fontsize=9)
        axes[1].text(rul[i] + 0.02, y[i], f"{rul[i]:.2f}", va="center", fontsize=9)
    axes[0].set_yticks(y, names, fontsize=9)
    axes[0].set_xlim(0, 1.15); axes[1].set_xlim(0, 1.15)
    axes[0].set_title("Can the flying condition be read from it?\n(want LOW)", loc="left", fontsize=10)
    axes[1].set_title("Can the remaining life be read from it?\n(want HIGH)", loc="left", fontsize=10)
    for a in axes:
        a.set_xlabel("R² of a probe on unseen engines")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "11b_koopman_control_probe.png")


# ---------------------------------------------------------------- 12. final results
def final_results():
    ds = ["FD001", "FD002", "FD003", "FD004"]
    vals = {"Paper's KePIN": [12.92, 15.08, 11.42, 17.04], "Original KePIN (our run)": [12.92, 15.00, 11.56, 17.11],
            "+ condition-aware input": [13.41, 14.13, 11.17, 15.32],
            "+ condition-aware + control": [12.85, 13.87, 11.07, 15.41]}
    cols = [MUTED, BLUE, GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    x = np.arange(4)
    for i, (k, v) in enumerate(vals.items()):
        ax.bar(x + (i - 1.5) * 0.2, v, 0.19, color=cols[i], label=k)
        for j in range(4):
            ax.text(x[j] + (i - 1.5) * 0.2, v[j] + 0.15, f"{v[j]:.2f}", ha="center", fontsize=7, rotation=90, va="bottom")
    ax.set_xticks(x, ["FD001\n(1 condition)", "FD002\n(6 conditions)", "FD003\n(1 condition)", "FD004\n(6 conditions)"])
    ax.set_ylim(8, 21)
    ax.set_ylabel("RMSE (ensemble of 3), lower = better")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    ax.set_title("Final accuracy on all four datasets: big gains where there are 6 flying conditions", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "12_final_results.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (kepin_architecture, ablation, dynamic_failure, fade_vs_grow, wear_architecture,
              condition_aware_architecture, condition_norm_real_data, koopman_control, final_results):
        f()
        print("ok", f.__name__)
