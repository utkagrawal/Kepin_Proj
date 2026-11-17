"""
Cross-domain datasets for the history-health hypernetwork study (original KePIN's multi-domain benchmark).

Windows and targets come from KePIN's own loader (GenericTimeSeriesDataset) with KePIN's own configs, so they
are the same windows the original / Signal A cross-domain runs used:
  Cylinder_Wake    150 simulated flows x 800 steps, 8 features, lift coefficient 5 steps ahead (unit split 80/20)
  Building_Energy  120 simulated buildings x 2160 steps, 10 features, energy use 6 h ahead (unit split 80/20)
  Jena_Climate     one hourly weather series (2009-2016), 14 features, temperature 24 h ahead (temporal split 80/20)
  SPY_Stock        one daily S&P 500 series, technical indicators, time-to-5%-drawdown target (temporal split 80/20)

For the history context we recover, for every window, its unit (series) and position: consecutive windows of
the same series overlap by seq_len-1 steps, so a series boundary is where that overlap breaks. Each series is
rebuilt from its windows, and the SAME history features as on C-MAPSS are computed from steps 1..t only:
  drift = mean of the last 10 steps - mean of the first 10 steps of the series; rate = slope over the last 30 steps
-> 2 x n_features numbers per window (features are already z-scored on the training split by KePIN's loader).
Single-series datasets (Jena, SPY) have no natural start, so their drift is measured from a trailing baseline
(10 windows back) instead; they are cut into 10 contiguous blocks only to define the clean-protocol validation split.
"""
import json
import os
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_DIR, "kepin_core")
sys.path.insert(0, _CORE)
_KCODE = os.path.abspath(os.path.join(_DIR, "..", "original_kepin", "code"))

DOMAINS = ["Cylinder_Wake", "Building_Energy", "Jena_Climate", "SPY_Stock"]
# training settings: Cylinder/Building/Jena as in the Signal A cross-domain runs (train_multidomain_conditioned.py),
# SPY as in original KePIN's run_optimized_multidomain.py. stride: use every k-th training window (heavily
# overlapping windows; keeps a run ~1 h). Test windows are never subsampled.
TRAIN_CFG = {
    "Cylinder_Wake":   dict(epochs=200, patience=50, lr=1e-3, batch_size=256, stride=2),
    "Building_Energy": dict(epochs=200, patience=50, lr=1e-3, batch_size=256, stride=4),
    "Jena_Climate":    dict(epochs=150, patience=40, lr=1e-3, batch_size=512, stride=1),
    "SPY_Stock":       dict(epochs=400, patience=80, lr=3e-4, batch_size=64, stride=1),
}
HIST_BASE, HIST_RECENT, HIST_SLOPE = 10, 10, 30


def _config(name):
    files = ["datasets_all_config.json", "datasets_multidomain_config.json"]
    for f in files:
        for c in json.load(open(os.path.join(_KCODE, f))):
            if c.get("name") == name:
                c = dict(c)
                for k in ("csv_path", "train_path", "test_path", "test_rul_path"):
                    if c.get(k) and not os.path.isabs(c[k]):
                        c[k] = os.path.join(_KCODE, c[k])
                return c
    raise KeyError(name)


def _segments(X):
    """Series id and 1-based position (steps seen up to the window end) for consecutive sliding windows."""
    cont = np.all(np.isclose(X[1:, :-1], X[:-1, 1:], atol=1e-6), axis=(1, 2))
    seg = np.concatenate([[0], np.cumsum(~cont)])
    pos = np.zeros(len(X), int)
    for s in np.unique(seg):
        i = np.where(seg == s)[0]
        pos[i] = np.arange(len(i)) + X.shape[1]
    return seg, pos


def _history(X, seg, pos, trailing=False):
    """trailing=True (one long series with an arbitrary start, e.g. weather, prices): the baseline is the mean of
    the 10 steps that start 10 windows back (10 * seq_len steps), instead of the series' first 10 steps."""
    T, F = X.shape[1], X.shape[2]
    t = np.arange(HIST_SLOPE) - (HIST_SLOPE - 1) / 2
    out = np.zeros((len(X), 2 * F), np.float32)
    for s in np.unique(seg):
        i = np.where(seg == s)[0]
        S = np.concatenate([X[i[0]], X[i[1:], -1]], 0).astype(np.float64)      # the whole series, (L, F)
        base = S[:HIST_BASE].mean(0)
        cs = np.concatenate([np.zeros((1, F)), np.cumsum(S, 0)], 0)
        e = pos[i]                                                               # steps seen (window end)
        recent = (cs[e] - cs[e - HIST_RECENT]) / HIST_RECENT
        if trailing:
            b0 = np.maximum(e - 10 * T, 0)
            base = (cs[b0 + HIST_BASE] - cs[b0]) / HIST_BASE
        win = np.lib.stride_tricks.sliding_window_view(S, HIST_SLOPE, axis=0)    # (L-29, F, 30)
        rate = 10 * (win @ t) / np.sum(t ** 2)                                   # slope ending at step k+30
        out[i, :F] = recent - base
        out[i, F:] = rate[e - HIST_SLOPE]
    return np.clip(out, -10, 10)


def load_domain(name):
    import GenericTimeSeriesDataset as GDS
    cfg = _config(name)
    cache = os.path.join(_DIR, "runs", "cache", f"domain_{name}_v1.npz")
    if os.path.exists(cache):
        f = np.load(cache)
        X_tr, Y_tr, X_te, Y_te = f["X_tr"], f["Y_tr"], f["X_te"], f["Y_te"]
    else:
        X_tr4, Y_tr, X_te4, Y_te = GDS.load_dataset_from_config(cfg).get_data()
        X_tr = X_tr4[:, :, 0, :] if X_tr4.ndim == 4 else X_tr4
        X_te = X_te4[:, :, 0, :] if X_te4.ndim == 4 else X_te4
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache + ".tmp.npz", X_tr=X_tr, Y_tr=Y_tr, X_te=X_te, Y_te=Y_te)
        os.replace(cache + ".tmp.npz", cache)
    X_tr, X_te = X_tr.astype(np.float32), X_te.astype(np.float32)
    Y_tr, Y_te = Y_tr.reshape(-1).astype(np.float32), Y_te.reshape(-1).astype(np.float32)
    seg_tr, pos_tr = _segments(X_tr)
    seg_te, pos_te = _segments(X_te)
    single = len(np.unique(seg_tr)) == 1
    hist_tr, hist_te = _history(X_tr, seg_tr, pos_tr, single), _history(X_te, seg_te, pos_te, single)
    units_tr = seg_tr.copy()
    if len(np.unique(seg_tr)) == 1:                       # one series: 10 contiguous blocks for the clean split
        units_tr = np.minimum(np.arange(len(X_tr)) * 10 // len(X_tr), 9)
    age_scale = float(pos_tr.max())                       # so that age/100 in train.py lies in (0, 1]
    n = lambda a: np.zeros((len(a), a.shape[1], 1), np.float32)
    return dict(X_train=X_tr, Y_train=Y_tr, X_test=X_te, Y_test=Y_te, X_train_raw=X_tr, X_test_raw=X_te,
                ema_alpha=None, units_train=units_tr, U_train=n(X_tr), U_test=n(X_te),
                age_train=100.0 * pos_tr / age_scale, age_test=100.0 * pos_te / age_scale,
                hist_train=hist_tr, hist_test=hist_te, n_conditions=1, ds_name=name,
                n_series_train=int(len(np.unique(seg_tr))), n_series_test=int(len(np.unique(seg_te))))


if __name__ == "__main__":
    from sklearn.linear_model import RidgeCV
    for d in (sys.argv[1:] or DOMAINS):
        D = load_domain(d)
        r = RidgeCV(alphas=np.logspace(-2, 3, 11))
        r2_h = r.fit(D["hist_train"], D["Y_train"]).score(D["hist_test"], D["Y_test"])
        last = lambda X: X[:, -1]
        r2_w = r.fit(last(D["X_train"]), D["Y_train"]).score(last(D["X_test"]), D["Y_test"])
        print(f"{d}: train {D['X_train'].shape} ({D['n_series_train']} series)  test {D['X_test'].shape} "
              f"({D['n_series_test']} series)  y range [{D['Y_train'].min():.3g}, {D['Y_train'].max():.3g}]  "
              f"test R2 from history {r2_h:.2f}, from last window step {r2_w:.2f}", flush=True)
