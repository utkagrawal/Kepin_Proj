#!/usr/bin/env python3
"""Overview figure: ensemble RMSE change vs original KePIN, paper protocol, all experiments."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(_DIR, "..", "runs", "paper")
ARMS = [("none_static", "KePIN, Koopman removed"),
        ("eig_static_wear", "Wear model (Koopman is the predictor)"),
        ("svd_static_future1", "KePIN + Koopman trained on real future"),
        ("svd_static_future1_sub8", "  …same, 16 dims + growth term"),
        ("svd_static_cn", "KePIN + condition-aware input"),
        ("svd_static_cn_ctrl", "KePIN + condition-aware input + Koopman with control")]
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
COLORS = {"FD001": "#2a78d6", "FD002": "#eb6834", "FD003": "#1baf7a", "FD004": "#6250d6"}


def ens(fd, arm):
    fs = sorted(glob.glob(os.path.join(RUNS, fd, arm, "run*_predictions.npz")))
    if len(fs) < 3:
        return None
    p = [np.load(f) for f in fs]
    return float(np.sqrt(np.mean((p[0]["y_true"] - np.mean([q["y_pred"] for q in p], 0)) ** 2)))


def main():
    base = {fd: ens(fd, "svd_static") for fd in DATASETS}
    table = {"KePIN (original, our run)": base}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    y = np.arange(len(ARMS))[::-1]
    off = {fd: (i - 1.5) * 0.17 for i, fd in enumerate(DATASETS)}
    for yi, (arm, label) in zip(y, ARMS):
        row = {}
        for fd in DATASETS:
            e = ens(fd, arm)
            row[fd] = e
            if e is None:
                continue
            d = e - base[fd]
            ax.barh(yi + off[fd], d, height=0.15, color=COLORS[fd], label=fd if yi == y[-1] or arm == "none_static" else None)
        table[label] = row
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_yticks(y, [l for _, l in ARMS])
    ax.set_xlabel("change in RMSE vs original KePIN (ensemble of 3, paper method)  ←  better | worse  →")
    ax.set_title("What helped and what did not (bars missing = not run on that dataset)", loc="left", fontsize=11)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hi, li in zip(h, l):
        seen.setdefault(li, hi)
    ax.legend([seen[k] for k in DATASETS if k in seen], [k for k in DATASETS if k in seen], frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(_DIR, "overview_vs_kepin.png")
    fig.savefig(out, dpi=160)
    json.dump(table, open(os.path.join(_DIR, "overview_vs_kepin.json"), "w"), indent=2)
    for k, v in table.items():
        print(f"{k:55s} " + "  ".join(f"{fd}={v[fd]:.2f}" if v[fd] is not None else f"{fd}=  -  " for fd in DATASETS))


if __name__ == "__main__":
    main()
