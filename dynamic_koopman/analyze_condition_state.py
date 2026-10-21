#!/usr/bin/env python3
"""
Does "Koopman with control" really separate flight conditions from wear? (FD002, paper runs 0-2)

For each trained model we take the hidden state the RUL head reads, pooled over the window
([mean_t, max_t], as the head does), on a sample of training windows, and ask probes how well
it predicts
  (a) the operating-condition mix of the window (6 fractions)  -> should be LOW for the wear state s
  (b) the true RUL                                             -> should stay HIGH
Probes: ridge regression (linear) and a random forest (non-linear), out-of-fold, folds = engines.

States compared:
  kepin_z   : original KePIN (svd_static) latent
  cn_z      : condition-aware input only (svd_static_cn) latent
  ctrl_z    : condition-aware + control model, latent BEFORE removing C*u
  ctrl_s    : same model, the wear state s = z - C*u that its head and Koopman losses use
Output: reports/fd002_condition_state/summary.json
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

OUT = os.path.join(_DIR, "reports", "fd002_condition_state")
RUNS = os.path.join(_DIR, "runs", "paper", "FD002")


def arch():
    a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
    a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
    return a


def build(kind, run, D, n_feat):
    if kind == "kepin":
        m = DynamicKePIN((30, n_feat), arch_config=arch(), koopman="svd", koopman_mode="static", head="kepin")
        m(tf.constant(D["X_test"][:2]), training=False)
        m.load_weights(os.path.join(RUNS, "svd_static", f"run{run}.weights.h5"))
        return m
    ctrl = kind == "ctrl"
    dummy = (np.zeros((6, n_feat), np.float32), np.ones((6, n_feat), np.float32))
    m = DynamicKePIN((30, n_feat), arch_config=arch(), koopman="svd", koopman_mode="static", head="kepin",
                     exo_dim=6, cond_norm=True, koop_control=ctrl, cond_stats=dummy, ema_alpha=D["ema_alpha"])
    U = D["U_test"][:2].reshape(2, -1)
    m((tf.constant(D["X_test_raw"][:2]), tf.constant(U)), training=False)
    m.load_weights(os.path.join(RUNS, "svd_static_cn_ctrl" if ctrl else "svd_static_cn", f"run{run}.weights.h5"))
    return m


def states(m, kind, X_ema, X_raw, U, bs=512):
    """Pooled hidden state(s) the RUL head sees: {name: (N, 2d)}."""
    out = {}
    for i in range(0, len(X_raw), bs):
        if kind == "kepin":
            z = m.encode(X_ema[i:i + bs], training=False)
            parts = {"z": z}
        else:
            x = tf.constant(X_raw[i:i + bs])
            u = tf.reshape(tf.constant(U[i:i + bs]), [-1, 30, 6])
            x = x * tf.einsum("btc,cf->btf", u, m.cn_gamma) + tf.einsum("btc,cf->btf", u, m.cn_beta)
            x = tf.einsum("ts,bsf->btf", m.ema_matrix, x)
            z = tf.cast(m.encode(x, training=False), tf.float32)
            parts = {"z": z}
            if kind == "ctrl":
                parts["s"] = z - tf.einsum("btc,cd->btd", u, m.control_C)
        for k, v in parts.items():
            pooled = tf.concat([tf.reduce_mean(v, 1), tf.reduce_max(v, 1)], -1).numpy()
            out.setdefault(k, []).append(pooled)
    return {k: np.concatenate(v) for k, v in out.items()}


def r2(y, p):
    y, p = np.atleast_2d(y.T).T, np.atleast_2d(p.T).T
    return float(np.mean([1 - ((y[:, j] - p[:, j]) ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum()
                          for j in range(y.shape[1])]))


def probes(F, mix, rul, groups):
    cv = GroupKFold(5)
    lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=0)
    return {"condition_R2_linear": r2(mix, cross_val_predict(lin, F, mix, groups=groups, cv=cv)),
            "condition_R2_forest": r2(mix, cross_val_predict(rf, F, mix, groups=groups, cv=cv)),
            "RUL_R2_linear": r2(rul, cross_val_predict(lin, F, rul, groups=groups, cv=cv)),
            "RUL_R2_forest": r2(rul, cross_val_predict(rf, F, rul, groups=groups, cv=cv))}


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_cmapss("FD002")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(D["X_train"]), 6000, replace=False))
    X_ema, X_raw = D["X_train"][idx], D["X_train_raw"][idx]
    U = D["U_train"][idx].reshape(len(idx), -1)
    mix = D["U_train"][idx].mean(axis=1)                  # condition fractions in each window
    rul = D["Y_train"].ravel()[idx]
    groups = D["units_train"][idx]
    n_feat = X_raw.shape[2]

    res = {}
    for kind, names in (("kepin", {"z": "kepin_z"}), ("cn", {"z": "cn_z"}), ("ctrl", {"z": "ctrl_z", "s": "ctrl_s"})):
        for run in range(3):
            m = build(kind, run, D, n_feat)
            st = states(m, kind, X_ema, X_raw, U)
            for k, name in names.items():
                r = probes(st[k], mix, rul, groups)
                res.setdefault(name, []).append(r)
                print(f"{name:8s} run{run}: " + "  ".join(f"{a}={b:.3f}" for a, b in r.items()), flush=True)
            tf.keras.backend.clear_session()
    summary = {name: {k: float(np.mean([r[k] for r in rs])) for k in rs[0]} for name, rs in res.items()}
    summary["per_run"] = res
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\nmean over 3 runs:")
    for name in ("kepin_z", "cn_z", "ctrl_z", "ctrl_s"):
        print(f"  {name:8s} " + "  ".join(f"{k}={v:.3f}" for k, v in summary[name].items()))


if __name__ == "__main__":
    main()
