#!/usr/bin/env python3
"""Figures for the hypernetwork explainability chapter and the all-datasets results (KePIN_Project_Explained.pdf).
Reads reports/hyper_explain/<ds>_clean.{npz,json} (analyze_hyper_explain.py) and runs/<protocol>/<ds>/... predictions."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_figures import (BLUE, GREEN, GRID, INK, MUTED, ORANGE, OUT, RED, VIOLET, YELLOW, arrow, box, canvas, save)

DK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EX = os.path.join(DK, "reports", "hyper_explain")
DSS = ["FD001", "FD002", "FD003", "FD004"]
DCOL = {"FD001": BLUE, "FD002": ORANGE, "FD003": GREEN, "FD004": VIOLET}


def load(ds):
    return np.load(os.path.join(EX, f"{ds}_clean.npz")), json.load(open(os.path.join(EX, f"{ds}_clean.json")))


def style(a):
    a.grid(color=GRID); a.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------- E1. the hypernetwork as a "settings generator"
def hyper_concept():
    fig, ax = canvas(11, 4.2, 11, 4.4)
    box(ax, 0.1, 2.9, 2.2, 1.2, "Engine's history\n(all past flights)", "violet")
    box(ax, 0.1, 1.2, 2.2, 1.2, "Health number h\n1 = healthy\n0 = near failure", "violet", bold=True)
    arrow(ax, 1.2, 2.9, 1.2, 2.4, VIOLET)
    box(ax, 2.9, 1.2, 2.4, 1.2, "HYPERNETWORK\n(a small network that\nwrites settings)", "orange", bold=True)
    arrow(ax, 2.3, 1.8, 2.9, 1.8, VIOLET)
    box(ax, 5.9, 1.2, 2.4, 1.2, "64 'fade speeds' ρ\n64 'turn speeds' θ\nfor THIS engine", "orange")
    arrow(ax, 5.3, 1.8, 5.9, 1.8, ORANGE)
    box(ax, 8.9, 1.2, 2.0, 1.2, "This engine's\nKoopman rule K", "orange", bold=True)
    arrow(ax, 8.3, 1.8, 8.9, 1.8, ORANGE)
    box(ax, 8.9, 2.9, 2.0, 1.2, "Shared 'directions' P\n(same for every\nengine)", "grey")
    arrow(ax, 9.9, 2.9, 9.9, 2.4, MUTED)
    ax.text(0.1, 0.55, "Analogy: P is the same set of 64 dials on every engine's control panel. "
            "The hypernetwork reads the health number\nand turns each dial (how fast that pattern fades, how fast it "
            "turns). A healthy engine and a failing engine get different dial settings.",
            fontsize=9.2, color=INK, va="center")
    save(fig, "22_hyper_concept.png")


# ---------------------------------------------------------------- E2. eigenvalues healthy vs near failure
def eig_circle():
    d, _ = load("FD002")
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.3))
    t = np.linspace(0, 2 * np.pi, 300)
    for a, key, col, title in [(ax[0], "eig_early", BLUE, "healthy (%d flights before failure)"),
                               (ax[1], "eig_late", ORANGE, "near failure (%d flights before failure)")]:
        e = d[key]
        a.plot(np.cos(t), np.sin(t), color=MUTED, lw=1)
        a.plot(0.3 * np.cos(t), 0.3 * np.sin(t), color=MUTED, lw=0.8, ls=":")
        a.scatter(e.real, e.imag, s=22, color=col, edgecolor="white", lw=0.5, zorder=3)
        a.scatter(e.real, -e.imag, s=22, color=col, edgecolor="white", lw=0.5, zorder=3, alpha=0.45)
        n_slow = int((np.abs(e) > 0.3).sum())
        a.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), aspect="equal", xlabel="real part", ylabel="imaginary part",
              title=(title % int(d[key + "_rem"])) + f"\n{n_slow} of 64 modes outside the dotted circle (ρ > 0.3)")
        a.title.set_fontsize(9.5)
        style(a)
    plt.suptitle("The same engine's generated Koopman rule: every dot is one mode (distance from centre = ρ)",
                 fontsize=10.5, x=0.02, ha="left")
    plt.tight_layout()
    save(fig, "23_eig_circle.png")


# ---------------------------------------------------------------- E3. the health dial (what-if)
def health_dial():
    d, js = load("FD002")
    flip = js["runs"][0]["h_flipped"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = [BLUE, GREEN, YELLOW, ORANGE, RED]
    for j in range(d["dial_rul"].shape[1]):
        ax[0].plot(d["dial_h"], d["dial_rul"][:, j], color=cols[j], lw=2,
                   label=f"engine with true RUL {int(d['dial_true'][j])}")
        ax[0].scatter([d["dial_hreal"][j]], [np.interp(d["dial_hreal"][j], d["dial_h"], d["dial_rul"][:, j])],
                      color=cols[j], edgecolor=INK, s=40, zorder=4)
    ax[0].set(xlabel="health number h we feed in (0 = near failure, 1 = healthy)", ylabel="predicted RUL (flights)",
              title="Turn the health dial: the prediction follows")
    ax[0].legend(frameon=False, fontsize=7.5)
    ax[1].plot(d["dial_h"], d["dial_slow"].mean(1), color=ORANGE, lw=2.4)
    ax[1].set(xlabel="health number h we feed in", ylabel="slow modes (ρ > 0.3) of 64",
              title="...and the Koopman rule changes shape")
    for a in ax:
        a.title.set_fontsize(10); style(a)
    if flip:
        fig.text(0.01, -0.02, "note: in this run h is flipped (higher = closer to failure)", fontsize=8, color=RED)
    plt.tight_layout()
    save(fig, "24_health_dial.png")


# ---------------------------------------------------------------- E4. life curves on all datasets
def life_all():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.7))
    for ds in DSS:
        d, js = load(ds)
        x = -d["bins"]
        for key, a in [("h", ax[0]), ("rho", ax[1]), ("slow", ax[2])]:
            ys = np.array([d[f"r{r}_{key}"] for r in range(3)])
            if key == "h":
                ys = np.array([y if not js["runs"][r]["h_flipped"] else 1 - y for r, y in enumerate(ys)])
            a.plot(x, np.nanmean(ys, 0), color=DCOL[ds], lw=2, label=ds)
            a.fill_between(x, np.nanmin(ys, 0), np.nanmax(ys, 0), color=DCOL[ds], alpha=0.15, lw=0)
    ax[0].set(ylabel="health number h", title="1. Health falls over life")
    ax[1].set(ylabel="average fade factor ρ per step", title="2. Koopman modes fade less near failure")
    ax[2].set(ylabel="slow modes (ρ > 0.3) of 64", title="3. More modes are kept near failure")
    for a in ax:
        a.set_xlabel("flights before failure"); a.title.set_fontsize(10); style(a)
    ax[0].legend(frameon=False, fontsize=8.5)
    plt.tight_layout()
    save(fig, "25_life_all.png")


# ---------------------------------------------------------------- E5. per-engine health trajectories
def h_trajectories():
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for a, ds in zip(ax, DSS):
        d, js = load(ds)
        flip = js["runs"][0]["h_flipped"]
        for j in range(6):
            if f"traj{j}_h" not in d:
                continue
            h = d[f"traj{j}_h"]
            a.plot(-d[f"traj{j}_rem"], 1 - h if flip else h, lw=1.4, alpha=0.85)
        share = np.mean([r["h_falls_over_life_share"] for r in js["runs"]])
        a.set(title=f"{ds}: h falls in {100 * share:.0f}% of engines", xlabel="flights before failure")
        a.title.set_fontsize(9.5); style(a)
    ax[0].set_ylabel("health number h")
    plt.tight_layout()
    save(fig, "26_h_trajectories.png")


# ---------------------------------------------------------------- E6. knockout per dataset
def knockout_all():
    keys = ["same K for all engines", "spectral features fixed", "h shuffled"]
    lab = ["same Koopman rule\nfor every engine", "12 Koopman features\nheld fixed", "health numbers\nshuffled"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.2
    for i, ds in enumerate(DSS):
        _, js = load(ds)
        inc = [np.mean([r["knockout"][k] - r["knockout"]["normal"] for r in js["runs"]]) for k in keys]
        b = ax.bar(np.arange(3) + (i - 1.5) * w, inc, w * 0.92, color=DCOL[ds], label=ds)
        for rect, v in zip(b, inc):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.4, f"+{v:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(3), lab, fontsize=9)
    ax.set_ylabel("RMSE increase (fair test, avg of 3 runs)")
    ax.legend(frameon=False, fontsize=8.5, ncol=4)
    ax.set_title("Knockout of the health-only Koopman model (arm A): break the Koopman path and the error explodes",
                 loc="left", fontsize=10)
    style(ax)
    plt.tight_layout()
    save(fig, "27_knockout_all.png")


# ---------------------------------------------------------------- E7. per-mode heatmap (FD002 and FD004)
def mode_heatmaps():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, ds in zip(ax, ["FD002", "FD004"]):
        d, js = load(ds)
        g = d["mode_rho_by_h"]
        if js["runs"][0]["h_flipped"]:
            g = g[::-1]
        order = np.argsort(np.nanmedian(g[-3:], 0))
        im = a.imshow(g[:, order].T, aspect="auto", origin="lower", extent=[0, 1, 0, g.shape[1]], cmap="viridis",
                      vmin=0, vmax=1)
        a.set(xlabel="health number h (0 = near failure, 1 = healthy)", ylabel="mode (sorted)", title=f"{ds}: fade factor ρ of each mode")
        a.title.set_fontsize(10)
    plt.colorbar(im, ax=ax, label="ρ (1 = never fades)")
    save(fig, "28_mode_heatmaps.png")


# ---------------------------------------------------------------- E8. results on all datasets
def results_all():
    def ens(p, ds, arm):
        fs = sorted(glob.glob(os.path.join(DK, "runs", p, ds, arm, "run*_predictions.npz")))
        if not fs:
            return np.nan
        ds_ = [np.load(f) for f in fs]
        y = ds_[0]["y_true"].ravel(); pr = np.mean([x["y_pred"].ravel() for x in ds_], 0)
        return float(np.sqrt(np.mean((pr - y) ** 2)))
    arms = [("svd_static", "original KePIN", MUTED), ("eig_hyper_hh_mono", "A: Koopman from health only", BLUE),
            ("eig_hyper_hh_age_rank_mono", "B: Koopman from health + age", ORANGE),
            ("eig_static_hh_hctx_mono", "C: static Koopman, health + age to predictor", GREEN)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    out = {}
    for a, p, title in [(ax[0], "clean", "Fair test (held-out training engines pick the model)"),
                        (ax[1], "paper", "Paper's method (test set picks the model)")]:
        w = 0.2
        for i, (arm, lab, c) in enumerate(arms):
            v = [ens(p, ds, arm) for ds in DSS]
            out[(p, arm)] = v
            b = a.bar(np.arange(4) + (i - 1.5) * w, v, w * 0.92, color=c, label=lab)
            for rect, x in zip(b, v):
                a.text(rect.get_x() + rect.get_width() / 2, x + 0.15, f"{x:.1f}", ha="center", fontsize=6.8, rotation=90)
        a.set_xticks(range(4), DSS); a.set_ylim(8, 20.5); a.set_title(title, fontsize=10); style(a)
    ax[0].set_ylabel("ensemble RMSE (3 seeds), lower is better")
    ax[0].legend(frameon=False, fontsize=7.8, loc="upper left")
    plt.tight_layout()
    save(fig, "29_results_all.png")
    return out


if __name__ == "__main__":
    hyper_concept(); eig_circle(); health_dial(); life_all(); h_trajectories(); knockout_all(); mode_heatmaps()
    print(results_all())
