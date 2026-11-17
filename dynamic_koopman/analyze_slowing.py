#!/usr/bin/env python3
"""
Is the "slower fading near failure" of the per-engine Koopman fit real critical slowing down,
or only the drift inside the window? (FD002, paper runs 0-2 of eig_health_cn_ctrl_eng_fc)

Per-engine K fits each mode's eigenvalue on the window in closed form (1-D DMD):
    lam = sum_t c_{t+1} conj(c_t) / sum_t |c_t|^2,   c_t = a_t + i b_t (modal coordinates)
A window that sits away from zero (offset) or drifts (trend) gives |lam| -> 1 even for a
fast-fading system, so the fit is repeated after removing, per window and mode,
    raw       nothing
    demean    the window mean
    detrend   a straight line in time (mean + slope)
If |lam| still rises as RUL falls after detrending, the engine's own dynamics really slow down.

Also model-free: lag-1 autocorrelation (the classic slowing-down indicator) of the
per-condition-normalised sensors over each 30-flight window, raw and detrended, and the
window variance (the second classic indicator).

Statistics on the 253 test engines (last window) and over life on all training windows
(Spearman of the window value against the flight index, per engine).
Output: reports/slowing/summary.json
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

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

RUNS = os.path.join(_DIR, "runs", "paper", "FD002", "eig_health_cn_ctrl_eng_fc")
OUT = os.path.join(_DIR, "reports", "slowing")
SENSORS_INFORMATIVE = ["s2", "s3", "s4", "s7", "s9", "s11", "s12", "s14", "s15", "s17", "s20", "s21"]
FEATS = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]


def build(run, D):
    a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
    a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
    dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))
    m = DynamicKePIN((30, 24), arch_config=a, koopman="eig", koopman_mode="health", exo_dim=6,
                     cond_norm=True, koop_control=True, cond_stats=dummy, ema_alpha=D["ema_alpha"],
                     adapt="engine", head_state=[1, 5, 10, 25, 50])
    m((tf.constant(D["X_test_raw"][:2]), tf.constant(D["U_test"][:2].reshape(2, -1))), training=False)
    m.load_weights(os.path.join(RUNS, f"run{run}.weights.h5"))
    return m


def modal(m, X, U, batch=2048):
    P = m.koopman.basis()
    out = []
    for i in range(0, len(X), batch):
        z, _ = m._condition_aware_latent((tf.constant(X[i:i + batch]),
                                          tf.constant(U[i:i + batch].reshape(len(X[i:i + batch]), -1))), False)
        out.append(tf.matmul(z, P).numpy())
    return np.concatenate(out)                                        # (N, T, d)


def remove(x, how):
    """x: (N, T, k). Remove nothing / the window mean / a straight line, along time."""
    if how == "raw":
        return x
    x = x - x.mean(1, keepdims=True)
    if how == "detrend":
        t = np.arange(x.shape[1]) - (x.shape[1] - 1) / 2
        x = x - t[None, :, None] * (np.einsum("t,ntk->nk", t, x) / np.sum(t ** 2))[:, None]
    return x


def dmd_abs(zt, how):
    m = zt.shape[-1] // 2
    a, b = remove(zt[..., :m], how), remove(zt[..., m:], how)
    a0, b0, a1, b1 = a[:, :-1], b[:, :-1], a[:, 1:], b[:, 1:]
    den = (a0 ** 2 + b0 ** 2).sum(1) + 1e-9
    re, im = (a1 * a0 + b1 * b0).sum(1) / den, (b1 * a0 - a1 * b0).sum(1) / den
    return np.sqrt(re ** 2 + im ** 2)                                 # (N, m)


def lag1(x, how):
    x = remove(x, how)
    return (x[:, 1:] * x[:, :-1]).sum(1) / ((x ** 2).sum(1) + 1e-9)   # (N, k)


def stats(v, rul, units=None, flight=None):
    """v: (N,) one value per window."""
    r = {"spearman_with_rul": float(spearmanr(v, rul)[0]),
         "median_rul_gt100": float(np.median(v[rul > 100])),
         "median_rul_le20": float(np.median(v[rul <= 20]))}
    if units is not None:
        rho = np.array([spearmanr(flight[units == u], v[units == u])[0] for u in np.unique(units)])
        r["life_spearman_median"] = float(np.nanmedian(rho))
        r["life_share_rising"] = float(np.mean(rho > 0))
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_cmapss("FD002")
    Xtr, Utr, units, Ytr = D["X_train_raw"], D["U_train"][..., :6], D["units_train"], D["Y_train"].ravel()
    Xte, Ute, Yte = D["X_test_raw"], D["U_test"][..., :6], D["Y_test"].ravel()
    flight = np.zeros(len(units), int)
    for u in np.unique(units):
        idx = np.where(units == u)[0]
        flight[idx] = np.arange(len(idx))
    summary = {"model": {}, "model_free": {}}

    for run in range(3):
        m = build(run, D)
        res = {}
        for split, X, U, y, uu, ff in (("test", Xte, Ute, Yte, None, None),
                                       ("train", Xtr, Utr, Ytr, units, flight)):
            zt = modal(m, X, U)
            for how in ("raw", "demean", "detrend"):
                lam = dmd_abs(zt, how)                                # (N, m)
                per_mode = np.array([spearmanr(lam[:, k], y)[0] for k in range(lam.shape[1])])
                r = stats(lam.mean(1), y, uu, ff)
                r["modes_rising_toward_failure"] = int(np.sum(per_mode < 0))
                r["n_modes"] = int(lam.shape[1])
                res[f"{split}_{how}"] = r
                print(f"run{run} {split:5s} {how:8s} | mean|lam| RUL>100 {r['median_rul_gt100']:.3f} "
                      f"RUL<=20 {r['median_rul_le20']:.3f} | Spearman(|lam|, RUL) {r['spearman_with_rul']:+.2f} | "
                      f"modes slowing {r['modes_rising_toward_failure']}/{r['n_modes']}"
                      + (f" | over life: median rho {r['life_spearman_median']:+.2f}, rising in "
                         f"{100 * r['life_share_rising']:.0f}% engines" if uu is not None else ""), flush=True)
        summary["model"][f"run{run}"] = res
        tf.keras.backend.clear_session()

    # model-free: per-condition-normalised informative sensors
    xs, cs = Xtr.reshape(-1, 24), Utr.reshape(-1, 6).argmax(1)
    mu = np.stack([xs[cs == c].mean(0) for c in range(6)])
    sd = np.maximum(np.stack([xs[cs == c].std(0) for c in range(6)]), 1e-6)
    cols = [FEATS.index(s) for s in SENSORS_INFORMATIVE]
    for split, X, U, y, uu, ff in (("test", Xte, Ute, Yte, None, None), ("train", Xtr, Utr, Ytr, units, flight)):
        c = U.argmax(-1)
        xn = ((X - mu[c]) / sd[c])[..., cols]
        for how in ("raw", "detrend"):
            r = stats(lag1(xn, how).mean(1), y, uu, ff)
            summary["model_free"][f"{split}_lag1_{how}"] = r
            print(f"model-free {split:5s} lag-1 autocorr {how:8s} | RUL>100 {r['median_rul_gt100']:.3f} "
                  f"RUL<=20 {r['median_rul_le20']:.3f} | Spearman with RUL {r['spearman_with_rul']:+.2f}"
                  + (f" | over life rising in {100 * r['life_share_rising']:.0f}% engines" if uu is not None else ""),
                  flush=True)
        r = stats(remove(xn, "detrend").var(1).mean(1), y, uu, ff)
        summary["model_free"][f"{split}_variance_detrended"] = r
        print(f"model-free {split:5s} variance (detrended)    | RUL>100 {r['median_rul_gt100']:.3f} "
              f"RUL<=20 {r['median_rul_le20']:.3f} | Spearman with RUL {r['spearman_with_rul']:+.2f}", flush=True)
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
