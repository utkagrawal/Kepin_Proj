# -*- coding: utf-8 -*-
"""C-MAPSS loading for the dynamic-Koopman experiments.

Uses the original KePIN loader (GenericTimeSeriesDataset + EMA smoothing) so the
sensor windows are identical to the KePIN paper pipeline, and adds:
  - unit (engine) id of every training window   -> engine-level validation split
  - per-time-step operating-condition one-hot   -> exogenous input for mode="exo"
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_DIR, "kepin_core")
sys.path.insert(0, _CORE)

import numpy as np
import pandas as pd

import GenericTimeSeriesDataset as GDS
from kepin_model import convert_4d_to_3d
from kepin_training import apply_ema_smoothing

FD_INDEX = {"FD001": 0, "FD002": 1, "FD003": 2, "FD004": 3}
N_CONDITIONS = 6


def _read_raw(path, column_names):
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.iloc[:, :len(column_names)]
    df.columns = column_names
    return df


def _condition_ids(df):
    """Exact 6-way operating-condition label from the rounded settings."""
    key = list(zip(df["setting1"].round(0), df["setting2"].round(2), df["setting3"].round(0)))
    return key


def _windows_meta(df, seq_len, cond_map, last_only):
    """Unit id, per-step condition one-hot and age (flights flown at the window's last step)
    for each window, in GDS window order."""
    units, onehots, ages = [], [], []
    for uid in sorted(df["id"].unique()):
        u = df[df["id"] == uid].sort_values("cycle")
        n = len(u)
        if n < seq_len:
            continue
        cond = np.array([cond_map.get(k, 0) for k in _condition_ids(u)])
        oh = np.eye(N_CONDITIONS, dtype=np.float32)[cond]
        starts = [n - seq_len] if last_only else range(n - seq_len + 1)
        for i in starts:
            units.append(uid)
            onehots.append(oh[i:i + seq_len])
            ages.append(i + seq_len)
    return np.array(units), np.stack(onehots), np.array(ages, np.float32)


HIST_BASE, HIST_RECENT, HIST_SLOPE = 10, 10, 30   # flights: "new" baseline, recent level, recent slope
SENSORS = [f"s{i}" for i in range(1, 22)]


def _history_features(df, seq_len, cond_map, stats, last_only):
    """Engine-history features for every window (GDS window order), from flights 1..t only.

    Each sensor is z-scored with its OWN operating condition's mean/std (training statistics),
    so the features follow wear, not the flight condition (the Signal A lesson). Per sensor:
      drift = mean of the last HIST_RECENT flights - mean of the first HIST_BASE flights
      rate  = slope over the last HIST_SLOPE flights (per 10 flights)
    -> 42 numbers per window, clipped to [-10, 10].
    """
    mu, sd = stats
    t = np.arange(HIST_SLOPE) - (HIST_SLOPE - 1) / 2
    out = []
    for uid in sorted(df["id"].unique()):
        u = df[df["id"] == uid].sort_values("cycle")
        n = len(u)
        if n < seq_len:
            continue
        c = np.array([cond_map.get(k, 0) for k in _condition_ids(u)])
        S = (u[SENSORS].to_numpy(np.float64) - mu[c]) / sd[c]
        base = S[:HIST_BASE].mean(0)
        ends = [n] if last_only else range(seq_len, n + 1)
        for e in ends:
            drift = S[e - HIST_RECENT:e].mean(0) - base
            rate = 10 * (t @ S[e - HIST_SLOPE:e]) / np.sum(t ** 2)
            out.append(np.concatenate([drift, rate]))
    return np.clip(np.array(out, np.float32), -10, 10)


def load_cmapss(fd):
    cfg_all = json.load(open(os.path.join(_CORE, "datasets_kepin_config.json")))
    cfg = cfg_all[FD_INDEX[fd]]
    for key in ("train_path", "test_path", "test_rul_path"):
        cfg[key] = os.path.join(_CORE, cfg[key])
    seq_len = cfg["sequence_length"]

    ds = GDS.load_dataset_from_config(cfg)
    X_tr4, Y_tr, X_te4, Y_te = ds.get_data()
    X_tr_raw, X_te_raw = convert_4d_to_3d(X_tr4), convert_4d_to_3d(X_te4)   # before EMA smoothing
    X_tr, alpha = apply_ema_smoothing(X_tr_raw)
    X_te, _ = apply_ema_smoothing(X_te_raw, alpha=alpha)

    raw_tr = _read_raw(cfg["train_path"], cfg["column_names"])
    raw_te = _read_raw(cfg["test_path"], cfg["column_names"])
    keys = sorted(set(_condition_ids(raw_tr)))
    # single-condition datasets have only noise-level variation in the settings -> 1 condition
    cond_map = {k: i for i, k in enumerate(keys)} if len(keys) == N_CONDITIONS else {}
    units_tr, U_tr, age_tr = _windows_meta(raw_tr, seq_len, cond_map, last_only=False)
    units_te, U_te, age_te = _windows_meta(raw_te, seq_len, cond_map, last_only=True)
    # per-condition sensor statistics of the training data, for the history features
    c_tr = np.array([cond_map.get(k, 0) for k in _condition_ids(raw_tr)])
    S_tr = raw_tr[SENSORS].to_numpy(np.float64)
    nc = max(len(cond_map), 1)
    hmu = np.stack([S_tr[c_tr == c].mean(0) for c in range(nc)])
    hsd = np.maximum(np.stack([S_tr[c_tr == c].std(0) for c in range(nc)]), 1e-3)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "cache", f"hist_{fd}_v1.npz")
    if os.path.exists(cache):
        f = np.load(cache)
        hist_tr, hist_te = f["tr"], f["te"]
    else:
        hist_tr = _history_features(raw_tr, seq_len, cond_map, (hmu, hsd), last_only=False)
        hist_te = _history_features(raw_te, seq_len, cond_map, (hmu, hsd), last_only=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache + ".tmp.npz", tr=hist_tr, te=hist_te)
        os.replace(cache + ".tmp.npz", cache)
    assert len(units_tr) == len(X_tr) and len(units_te) == len(X_te), "window alignment mismatch"

    return dict(X_train=X_tr.astype(np.float32), Y_train=Y_tr.astype(np.float32),
                X_test=X_te.astype(np.float32), Y_test=Y_te.astype(np.float32),
                X_train_raw=X_tr_raw.astype(np.float32), X_test_raw=X_te_raw.astype(np.float32),
                ema_alpha=float(alpha),
                units_train=units_tr, U_train=U_tr, U_test=U_te,
                age_train=age_tr, age_test=age_te,      # flights flown so far (known at test time)
                hist_train=hist_tr, hist_test=hist_te,  # engine-history features (42), past flights only
                n_conditions=len(cond_map) or 1, ds_name=cfg["name"])


def engine_split(units, val_frac=0.2, seed=0):
    """Boolean masks (train, val) holding out whole engines."""
    rng = np.random.default_rng(seed)
    ids = np.unique(units)
    val_ids = rng.choice(ids, int(round(len(ids) * val_frac)), replace=False)
    val = np.isin(units, val_ids)
    return ~val, val
