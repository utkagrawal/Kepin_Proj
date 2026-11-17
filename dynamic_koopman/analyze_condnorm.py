#!/usr/bin/env python3
"""
Why does the condition-aware input layer help?  (FD002, FD004)

A. Signal analysis on the data (backs the mathematical argument):
   model each sensor as  x_t = m(c_t) + d_t + e_t   (condition offset + slow wear + noise).
   - eta2: share of a sensor's variance explained by the flying condition alone (between-condition variance).
   - |corr(x, RUL)| under one global scaling (KePIN) vs per-condition scaling.
   - order of operations: last-step EMA output for  (i) EMA of globally scaled data  vs
     (ii) EMA of per-condition scaled data; how much condition variance is left and how
     well it tracks RUL. Theory for random condition switching: EMA leaves a fraction
     alpha/(2-alpha) of the condition variance (~3% for alpha = 2/31) but mixes conditions.
B. Parameter analysis of the trained models (paper protocol, runs 0-2):
   learned per-condition scale gamma[c,f] and offset beta[c,f] vs their initial values
   (per-condition standardisation computed exactly as in train.py), and which sensors
   training amplified or suppressed.
Output: reports/condnorm_analysis/{summary.json, *.png}
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dk_data import load_cmapss

OUT = os.path.join(_DIR, "reports", "condnorm_analysis")
TEAL, ORANGE, GREY = "#0F6E76", "#D9822B", "#9A9A9A"


def ema_last(x, alpha):
    """KePIN's EMA (y0 = x0) applied along time; returns the last step. x: (N, T, F)."""
    y = x[:, 0].copy()
    for t in range(1, x.shape[1]):
        y = alpha * x[:, t] + (1 - alpha) * y
    return y


def eta2(v, c, k):
    """Share of variance of v explained by condition label c (between-group variance / total)."""
    mu = v.mean()
    between = sum((c == j).sum() * (v[c == j].mean() - mu) ** 2 for j in range(k) if (c == j).any())
    return float(between / (((v - mu) ** 2).sum() + 1e-12))


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 else 0.0


def main():
    os.makedirs(OUT, exist_ok=True)
    feats_all = json.load(open(os.path.join(_DIR, "kepin_core", "datasets_kepin_config.json")))
    summary = {}
    for fd, idx in (("FD002", 1), ("FD004", 3)):
        names = feats_all[idx]["feature_cols"]
        D = load_cmapss(fd)
        X = D["X_train_raw"].astype(np.float64)            # (N, 30, F) globally standardised, not smoothed
        U = D["U_train"]                                    # (N, 30, 6) one-hot condition per step
        y = D["Y_train"].ravel()
        alpha = D["ema_alpha"]
        k = U.shape[-1]
        cond = U.argmax(-1)                                 # (N, 30)
        F = X.shape[2]

        # per-condition statistics exactly as train.py (all training time steps, std floor 0.1)
        xs, cs = X.reshape(-1, F), cond.reshape(-1)
        mu = np.stack([xs[cs == j].mean(0) for j in range(k)])
        sd = np.maximum(np.stack([xs[cs == j].std(0) for j in range(k)]), 0.1)
        Xc = (X - mu[cond]) / sd[cond]                      # per-condition scaled windows

        # ---- A. signal analysis (last time step of each window = one flight)
        g_last, c_last, cl = X[:, -1], Xc[:, -1], cond[:, -1]
        ema_g, ema_c = ema_last(X, alpha), ema_last(Xc, alpha)   # (i) KePIN order, (ii) new order
        per_sensor = []
        for f in range(F):
            per_sensor.append({
                "sensor": names[f],
                "eta2_condition_global": eta2(g_last[:, f], cl, k),
                "eta2_condition_percond": eta2(c_last[:, f], cl, k),
                "absorr_rul_global": abs(corr(g_last[:, f], y)),
                "absorr_rul_percond": abs(corr(c_last[:, f], y)),
                "absorr_rul_ema_global_first": abs(corr(ema_g[:, f], y)),
                "absorr_rul_ema_percond_first": abs(corr(ema_c[:, f], y)),
            })
        ps = per_sensor
        A = {
            "mean_eta2_condition_global": float(np.mean([p["eta2_condition_global"] for p in ps])),
            "mean_eta2_condition_percond": float(np.mean([p["eta2_condition_percond"] for p in ps])),
            "mean_absorr_rul_global": float(np.mean([p["absorr_rul_global"] for p in ps])),
            "mean_absorr_rul_percond": float(np.mean([p["absorr_rul_percond"] for p in ps])),
            "mean_absorr_rul_ema_global_first": float(np.mean([p["absorr_rul_ema_global_first"] for p in ps])),
            "mean_absorr_rul_ema_percond_first": float(np.mean([p["absorr_rul_ema_percond_first"] for p in ps])),
            "max_absorr_rul_global": float(max(p["absorr_rul_global"] for p in ps)),
            "max_absorr_rul_percond": float(max(p["absorr_rul_percond"] for p in ps)),
            "ema_alpha": alpha,
            "theory_condition_variance_left_by_ema": alpha / (2 - alpha),
            "per_sensor": ps,
        }

        # ---- B. parameter analysis
        B = {"runs": []}
        gammas, betas = [], []
        for run in range(3):
            f = h5py.File(os.path.join(_DIR, "runs", "paper", fd, "svd_static_cn", f"run{run}.weights.h5"), "r")
            g, b = f["vars/0"][()], f["vars/1"][()]
            gammas.append(g)
            betas.append(b)
            g0, b0 = 1.0 / sd, -mu / sd
            rel_g = np.abs(g - g0) / np.abs(g0)
            # equivalent learned normalisation: x' = (x - m_learned) / s_learned
            s_l, m_l = 1.0 / g, -b / g
            B["runs"].append({
                "run": run,
                "median_rel_change_scale": float(np.median(rel_g)),
                "p90_rel_change_scale": float(np.percentile(rel_g, 90)),
                "median_abs_change_offset": float(np.median(np.abs(b - b0))),
                "corr_learned_vs_init_scale": corr(g.ravel(), g0.ravel()),
                "corr_learned_vs_init_offset": corr(b.ravel(), b0.ravel()),
                "median_shift_of_centre_in_sd_units": float(np.median(np.abs(m_l - mu) / sd)),
            })
        G = np.mean(gammas, 0) / (1.0 / sd)                  # learned scale relative to init, avg over runs
        gain = G.mean(0)                                      # per sensor, averaged over conditions
        order = np.argsort(-gain)
        B["sensor_gain_vs_init_mean_over_conditions_and_runs"] = {names[i]: float(gain[i]) for i in order}
        B["gain_consistency_across_runs_corr"] = float(np.mean([
            corr((gammas[a] * sd).ravel(), (gammas[b] * sd).ravel()) for a, b in ((0, 1), (0, 2), (1, 2))]))
        summary[fd] = {"A_signal": A, "B_parameters": B}

        # ---- figures
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
        xi = np.arange(F)
        ax[0].bar(xi - 0.2, [p["eta2_condition_global"] for p in ps], 0.4, color=GREY, label="one global scaling (KePIN)")
        ax[0].bar(xi + 0.2, [p["eta2_condition_percond"] for p in ps], 0.4, color=TEAL, label="per-condition scaling")
        ax[0].set_title(f"{fd}: share of each sensor's variation due to the flying condition", loc="left", fontsize=11)
        ax[1].bar(xi - 0.2, [p["absorr_rul_global"] for p in ps], 0.4, color=GREY, label="one global scaling (KePIN)")
        ax[1].bar(xi + 0.2, [p["absorr_rul_percond"] for p in ps], 0.4, color=TEAL, label="per-condition scaling")
        ax[1].set_title(f"{fd}: how strongly each sensor tracks remaining life |corr|", loc="left", fontsize=11)
        for a in ax:
            a.set_xticks(xi, names, rotation=70, fontsize=8)
            a.set_ylim(0, 1.05)
            a.legend(frameon=False, fontsize=9)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{fd}_signal.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(xi, G.mean(0), color=TEAL)
        ax.errorbar(xi, G.mean(0), yerr=np.std([gm * sd for gm in gammas], axis=0).mean(0), fmt="none", ecolor="#222222", lw=1)
        ax.axhline(1, color="#222222", ls="--", lw=1)
        ax.set_xticks(xi, names, rotation=70, fontsize=8)
        ax.set_ylabel("learned scale / initial scale")
        ax.set_title(f"{fd}: how training changed each sensor's per-condition scale (1 = unchanged)", loc="left", fontsize=11)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{fd}_learned_scale.png"), dpi=150)
        plt.close(fig)

        print(f"\n== {fd} ==")
        for key in ("mean_eta2_condition_global", "mean_eta2_condition_percond", "mean_absorr_rul_global",
                    "mean_absorr_rul_percond", "max_absorr_rul_global", "max_absorr_rul_percond",
                    "mean_absorr_rul_ema_global_first", "mean_absorr_rul_ema_percond_first",
                    "theory_condition_variance_left_by_ema"):
            print(f"  {key:42s} {A[key]:.3f}")
        for r in B["runs"]:
            print("  run", r["run"], {k: round(v, 3) for k, v in r.items() if k != "run"})
        top = list(B["sensor_gain_vs_init_mean_over_conditions_and_runs"].items())
        print("  amplified most:", [(n, round(v, 2)) for n, v in top[:5]])
        print("  suppressed most:", [(n, round(v, 2)) for n, v in top[-5:]])
        print("  consistency of learned scales across runs (corr):", round(B["gain_consistency_across_runs_corr"], 3))
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
