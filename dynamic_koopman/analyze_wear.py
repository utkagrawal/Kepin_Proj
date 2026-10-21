#!/usr/bin/env python3
"""
Explainability checks for the wear-accumulator model (FD002, clean protocol, runs 0-2).

1. Life trajectories: on held-out validation engines (never trained on), run the model on
   every window from first flight to failure. Physically, the wear rate should rise and the
   predicted RUL should fall as the engine ages.
2. Shortcut test (same as the Signal A analysis): is the wear rate explained by RUL or by the
   mix of operating conditions in the window? Out-of-fold R^2, folds split by engine.
3. Attribution: Integrated Gradients of the wear rate w.r.t. the input sensors (test engines),
   summed over time -> which sensors drive the wear rate.

Outputs: reports/fd002_wear_xai/{summary.json, life_trajectories.png, wear_attribution.png}
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss, engine_split
from dk_model import DynamicKePIN

OUT = os.path.join(_DIR, "reports", "fd002_wear_xai")
RUN_DIR = os.path.join(_DIR, "runs", "clean", "FD002", "eig_static_wear")
INK, MUTED = "#0b0b0b", "#8a8984"
C1, C2 = "#2a78d6", "#eb6834"


def build(n_feat, run):
    arch = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
    arch["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in arch["kernels"]]]
    m = DynamicKePIN((30, n_feat), arch_config=arch, koopman="eig", koopman_mode="static", head="wear")
    m(tf.zeros((2, 30, n_feat)), training=False)
    m.load_weights(os.path.join(RUN_DIR, f"run{run}.weights.h5"))
    return m


def infer(m, X, bs=1024):
    rul, v, h0 = [], [], []
    for i in range(0, len(X), bs):
        r, ko = m(X[i:i + bs], training=False)
        rul.append(r.numpy().ravel())
        v.append(ko["wear_rate"].numpy().ravel())
        h0.append(ko["health_trajectory"][:, 0].numpy())
    return np.concatenate(rul), np.concatenate(v), np.concatenate(h0)


def wear_rate_fn(m):
    def f(x):
        _, ko = m(x, training=False)
        return tf.reshape(ko["wear_rate"], [-1])
    return f


def integrated_gradients(f, X, steps=32, bs=256):
    out = np.zeros_like(X)
    for i in range(0, len(X), bs):
        x = tf.constant(X[i:i + bs])
        total = tf.zeros_like(x)
        for a in (np.arange(steps) + 0.5) / steps:            # baseline = 0 (standardized mean)
            xa = a * x
            with tf.GradientTape() as t:
                t.watch(xa)
                y = f(xa)
            total += t.gradient(y, xa)
        out[i:i + bs] = (x * total / steps).numpy()
    return out


def oof_r2(X, y, groups):
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=0)
    pred = cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(5))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_cmapss("FD002")
    feats = json.load(open(os.path.join(_DIR, "kepin_core", "datasets_kepin_config.json")))[1]["feature_cols"]
    _, va = engine_split(D["units_train"], val_frac=0.2, seed=0)        # same split as training
    Xv, Yv, Uv = D["X_train"][va], D["Y_train"].ravel()[va], D["units_train"][va]
    mix_v = D["U_train"][va].mean(axis=1)                                 # condition mix per window
    summary = {"n_val_engines": int(len(np.unique(Uv))), "n_val_windows": int(len(Xv)), "runs": {}}
    attributions = []

    for run in range(3):
        m = build(Xv.shape[2], run)
        rul, v, h0 = infer(m, Xv)
        rs = {}
        # 1. life trajectories: within each engine, windows are in time order
        rho_v, rho_rul = [], []
        for u in np.unique(Uv):
            idx = np.where(Uv == u)[0]
            t = np.arange(len(idx))
            rho_v.append(spearmanr(t, v[idx])[0])
            rho_rul.append(spearmanr(t, rul[idx])[0])
        rho_v, rho_rul = np.array(rho_v), np.array(rho_rul)
        rs["life_wear_rate_trend_median_spearman"] = float(np.nanmedian(rho_v))
        rs["life_engines_wear_rate_rising"] = float(np.mean(rho_v > 0))
        rs["life_pred_rul_trend_median_spearman"] = float(np.nanmedian(rho_rul))
        rs["life_engines_pred_rul_falling"] = float(np.mean(rho_rul < 0))
        rs["val_rmse_all_windows"] = float(np.sqrt(np.mean((rul - Yv) ** 2)))
        rs["corr_wear_rate_vs_true_rul_val"] = float(np.corrcoef(v, Yv)[0, 1])
        # 2. shortcut test on the wear rate (and current health)
        sub = np.random.default_rng(0).choice(len(Xv), min(6000, len(Xv)), replace=False)
        for name, y in (("wear_rate", v), ("current_health", h0)):
            rs[f"R2_{name}_from_RUL"] = oof_r2(Yv[sub, None], y[sub], Uv[sub])
            rs[f"R2_{name}_from_condition_mix"] = oof_r2(mix_v[sub], y[sub], Uv[sub])
        # 3. attribution on test engines
        ig = integrated_gradients(wear_rate_fn(m), D["X_test"])
        imp = np.abs(ig).sum(axis=1).mean(axis=0)                         # per feature
        attributions.append(imp / imp.sum())
        rs["top_features"] = [feats[i] for i in np.argsort(-imp)[:8]]
        summary["runs"][run] = rs
        print(f"run {run}: " + json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in rs.items()}), flush=True)

        if run == 0:   # life-trajectory figure for 4 validation engines of different lengths
            engines = np.unique(Uv)
            lengths = np.array([np.sum(Uv == u) for u in engines])
            picks = engines[np.argsort(lengths)[np.linspace(5, len(engines) - 6, 4).astype(int)]]
            fig, axes = plt.subplots(2, 4, figsize=(15, 5.6), sharex="col")
            for j, u in enumerate(picks):
                idx = np.where(Uv == u)[0]
                age = np.arange(len(idx)) + 30
                axes[0, j].plot(age, Yv[idx], color=MUTED, lw=1.5, ls="--", label="true RUL")
                axes[0, j].plot(age, rul[idx], color=C1, lw=1.5, label="predicted RUL")
                axes[0, j].set_title(f"held-out engine {u}", loc="left", fontsize=10)
                axes[1, j].plot(age, v[idx], color=C2, lw=1.5)
                axes[1, j].set_xlabel("engine age (flights)")
                for a in axes[:, j]:
                    for s in ("top", "right"):
                        a.spines[s].set_visible(False)
            axes[0, 0].set_ylabel("remaining life (flights)")
            axes[1, 0].set_ylabel("learned wear rate")
            axes[0, 0].legend(frameon=False, fontsize=8)
            fig.suptitle("Over each engine's life the wear rate climbs and predicted life falls "
                         "(FD002, engines never seen in training)", x=0.01, ha="left", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "life_trajectories.png"), dpi=150)

    att = np.array(attributions)
    order = np.argsort(-att.mean(0))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(feats)), att.mean(0)[order], color=C1, width=0.7)
    ax.errorbar(np.arange(len(feats)), att.mean(0)[order], yerr=att.std(0)[order], fmt="none", ecolor=INK, lw=1)
    ax.set_xticks(np.arange(len(feats)), [feats[i] for i in order], rotation=60, fontsize=8)
    ax.set_ylabel("share of wear-rate attribution")
    ax.set_title("Which inputs drive the learned wear rate (Integrated Gradients, mean ± std over 3 runs)",
                 loc="left", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "wear_attribution.png"), dpi=150)
    summary["attribution_share_mean"] = {feats[i]: float(att.mean(0)[i]) for i in order}
    summary["attribution_share_settings"] = float(att.mean(0)[:3].sum())
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("settings share of attribution:", round(summary["attribution_share_settings"], 3))
    print("top 8 overall:", [feats[i] for i in order[:8]])


if __name__ == "__main__":
    main()
