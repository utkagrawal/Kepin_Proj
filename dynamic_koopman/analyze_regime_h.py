#!/usr/bin/env python3
"""
What drives the health index h of the regime-adaptive Koopman model? (FD002, paper runs 0-2)

h is computed as  sigmoid(MLP([mean_t z_t, z_T]))  from the encoder's latent sequence, which
is built from per-condition-normalised, smoothed sensors. This script asks:

1. What h encodes (the test that exposed Signal A). On all training windows, out-of-fold
   R^2 (folds = engines) of h predicted from
     - the window's operating-condition mix (6 fractions) + condition of its last flight
     - the true (capped) RUL
   with a linear probe (ridge) and a non-linear one (random forest).
2. Is h monotone over each engine's life?  Spearman(h, flight) per training engine.
3. Which sensors drive h: Integrated Gradients of h w.r.t. the raw input window, on the
   253 test engines. Baseline = each flight's operating-condition mean, i.e. the input
   that the condition-normalisation maps to zero ("an average engine in that condition").
   Shares of total |IG| per feature, per flight position, and the signed effect on the
   engines closest to failure. Completeness (sum IG = h(x) - h(baseline)) is reported.
4. Model-free check: correlation of each per-condition-normalised sensor with RUL.

Arms: mono (main) and sup (reference). Output: reports/regime_h/{summary.json, regime_h.png}
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

RUNS = os.path.join(_DIR, "runs", "paper", "FD002")
OUT = os.path.join(_DIR, "reports", "regime_h")
ARMS = {"mono": "eig_health_cn_ord_hs10_mono", "sup": "eig_health_cn_ord_hs10_sup"}
FEATS = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
IG_STEPS = 32


def arch():
    a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
    a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
    return a


def build(arm, run, D):
    dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))   # overwritten by load
    m = DynamicKePIN((30, 24), arch_config=arch(), koopman="eig", koopman_mode="health",
                     exo_dim=6, cond_norm=True, cond_stats=dummy, ema_alpha=D["ema_alpha"],
                     health_map="ordered", head_state=10)
    m((tf.constant(D["X_test_raw"][:2]), tf.constant(D["U_test"][:2].reshape(2, -1))), training=False)
    m.load_weights(os.path.join(RUNS, arm, f"run{run}.weights.h5"))
    return m


def health(m, X, U, batch=1024):
    out = []
    for i in range(0, len(X), batch):
        out.append(m.regime_health((tf.constant(X[i:i + batch]),
                                    tf.constant(U[i:i + batch].reshape(len(X[i:i + batch]), -1))),
                                   training=False).numpy().ravel())
    return np.concatenate(out)


def oof_r2(F, y, groups, kind):
    est = (RidgeCV(alphas=np.logspace(-3, 3, 13)) if kind == "linear"
           else RandomForestRegressor(n_estimators=100, min_samples_leaf=20, n_jobs=8, random_state=0))
    p = cross_val_predict(est, F, y, groups=groups, cv=GroupKFold(5))
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def integrated_gradients(m, X, U):
    """IG of h w.r.t. the raw window; baseline = per-step condition mean (normalises to 0)."""
    g, b = m.cn_gamma.numpy(), m.cn_beta.numpy()                       # (6, 24)
    base = np.einsum("btc,cf->btf", U, -b / g).astype(np.float32)      # x0 with x0*g + b = 0
    Uf = tf.constant(U.reshape(len(U), -1))
    x, x0 = tf.constant(X), tf.constant(base)
    total = tf.zeros_like(x)
    for a in (np.arange(IG_STEPS) + 0.5) / IG_STEPS:                  # midpoint Riemann sum
        xi = x0 + float(a) * (x - x0)          # python float: numpy scalar * tensor -> numpy
        with tf.GradientTape() as tape:
            tape.watch(xi)
            h = tf.reduce_sum(m.regime_health((xi, Uf), training=False))   # sum INSIDE the tape
        total += tape.gradient(h, xi)
    ig = ((x - x0) * total / IG_STEPS).numpy()                         # (B, 30, 24)
    h_x = m.regime_health((x, Uf), training=False).numpy().ravel()
    h_0 = m.regime_health((x0, Uf), training=False).numpy().ravel()
    gap = np.abs(ig.sum((1, 2)) - (h_x - h_0))
    return ig, h_x, h_0, float(np.median(gap)), float(np.median(np.abs(h_x - h_0)))


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_cmapss("FD002")
    Xtr, Utr, units, Ytr = D["X_train_raw"], D["U_train"], D["units_train"], D["Y_train"].ravel()
    Xte, Ute, Yte = D["X_test_raw"], D["U_test"], D["Y_test"].ravel()
    mix = Utr.mean(1)                                                  # (N, 6) condition mix
    cond_feats = np.concatenate([mix, Utr[:, -1]], axis=1)             # + last-flight condition
    flight = np.zeros(len(units), int)
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        flight[idx] = np.arange(len(idx))

    # model-free: per-condition-normalised last-flight sensor vs RUL
    xs, cs = Xtr.reshape(-1, 24), Utr.reshape(-1, 6).argmax(1)
    mu = np.stack([xs[cs == c].mean(0) for c in range(6)])
    sd = np.maximum(np.stack([xs[cs == c].std(0) for c in range(6)]), 0.1)
    last_c = Utr[:, -1].argmax(1)
    xn_last = (Xtr[:, -1] - mu[last_c]) / sd[last_c]
    sensor_corr = {f: float(np.corrcoef(xn_last[:, j], Ytr)[0, 1]) for j, f in enumerate(FEATS)
                   if xn_last[:, j].std() > 1e-6}

    summary = {"sensor_corr_rul_condnorm": sensor_corr, "arms": {}}
    keep = {}
    for name, arm in ARMS.items():
        rows = []
        for run in range(3):
            m = build(arm, run, D)
            h_tr = health(m, Xtr, Utr)
            r = {"run": run,
                 "r2_from_condition_linear": oof_r2(cond_feats, h_tr, units, "linear"),
                 "r2_from_condition_forest": oof_r2(cond_feats, h_tr, units, "forest"),
                 "r2_from_rul_linear": oof_r2(Ytr[:, None], h_tr, units, "linear"),
                 "r2_from_rul_forest": oof_r2(Ytr[:, None], h_tr, units, "forest"),
                 "corr_h_rul_train": float(np.corrcoef(h_tr, Ytr)[0, 1])}
            rho = np.array([spearmanr(flight[units == u], h_tr[units == u])[0] for u in np.unique(units)])
            r["life_spearman_median"] = float(np.nanmedian(rho))
            r["life_share_decreasing"] = float(np.mean(rho < 0))
            ig, h_x, h_0, gap, scale = integrated_gradients(m, Xte, Ute)
            share = np.abs(ig).sum((0, 1))
            share = share / share.sum()
            tshare = np.abs(ig).sum((0, 2))
            tshare = tshare / tshare.sum()
            near = Yte <= 30
            r.update(ig_completeness_median_error=gap, ig_median_h_change=scale,
                     h_baseline_median=float(np.median(h_0)),
                     ig_share={f: float(s) for f, s in zip(FEATS, share)},
                     ig_share_last5_flights=float(tshare[-5:].sum()),
                     ig_share_first15_flights=float(tshare[:15].sum()),
                     ig_signed_near_failure={f: float(v) for f, v in zip(FEATS, ig[near].sum(1).mean(0))})
            rows.append(r)
            print(f"{name} run{run}: R2 cond lin/rf {r['r2_from_condition_linear']:.3f}/"
                  f"{r['r2_from_condition_forest']:.3f} | R2 RUL lin/rf {r['r2_from_rul_linear']:.3f}/"
                  f"{r['r2_from_rul_forest']:.3f} | life rho {r['life_spearman_median']:.2f} "
                  f"({100 * r['life_share_decreasing']:.0f}% decreasing) | IG err {gap:.4f} of {scale:.3f}",
                  flush=True)
            top = sorted(r["ig_share"].items(), key=lambda kv: -kv[1])[:8]
            print("   top IG:", ", ".join(f"{k} {100 * v:.1f}%" for k, v in top), flush=True)
            if name == "mono":
                keep.setdefault("h_tr", []).append(h_tr)
                keep.setdefault("share", []).append(share)
                keep.setdefault("tshare", []).append(tshare)
            tf.keras.backend.clear_session()
        summary["arms"][name] = rows
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.savez(os.path.join(OUT, "mono_arrays.npz"), h_train=np.array(keep["h_tr"]),
             ig_share=np.array(keep["share"]), ig_time_share=np.array(keep["tshare"]),
             units=units, flight=flight, rul=Ytr)
    plot(keep, units, flight, Ytr)


def plot(keep, units, flight, Ytr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, blue = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink,
                         "xtick.color": ink2, "ytick.color": ink2, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="#fcfcfb")
    share = np.array(keep["share"]) * 100
    order = np.argsort(share.mean(0))[::-1][:12][::-1]
    y = np.arange(len(order))
    ax[0].barh(y, share.mean(0)[order], color=blue, height=0.6)
    for r in range(share.shape[0]):
        ax[0].plot(share[r, order], y, "o", ms=3, color=ink, alpha=0.7)
    ax[0].set_yticks(y, [FEATS[i] for i in order])
    ax[0].set_xlabel("share of |attribution| to h (%)")
    ax[0].set_title("What drives h (Integrated Gradients,\n12 largest; bar = mean, dots = runs)", fontsize=9)
    ax[0].grid(axis="x", color=grid)

    h = keep["h_tr"][0]
    hb = ax[1].hexbin(Ytr, h, gridsize=40, cmap="Blues", mincnt=1, bins="log")
    ax[1].set_xlabel("true RUL (flights, capped at 125)")
    ax[1].set_ylabel("health h")
    ax[1].set_title("h vs true RUL, all training windows (run 0)", fontsize=9)
    fig.colorbar(hb, ax=ax[1], label="windows (log)")

    rng = np.random.default_rng(0)
    for u in rng.choice(np.unique(units), 12, replace=False):
        idx = units == u
        ax[2].plot(flight[idx], h[idx], color=blue, lw=1, alpha=0.6)
    ax[2].set_xlabel("flight (window index within engine)")
    ax[2].set_ylabel("health h")
    ax[2].set_title("h over life, 12 random training engines (run 0)", fontsize=9)
    ax[2].grid(color=grid)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regime_h.png"), dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    main()
