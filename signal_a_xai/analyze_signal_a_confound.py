#!/usr/bin/env python3
"""
Confound test for Signal A on FD002: is the condition network reacting to
degradation (RUL) or to the operating-condition mix inside each 30-cycle window?

FD002 cycles randomly through 6 operating conditions and sensors are standardized
globally, so window moments can be driven by condition switching. For each training
window we rebuild the operating-condition composition (6 fractions) and compare
out-of-fold R^2 of a random forest predicting Signal A / delta_s from
  (a) RUL only, (b) condition composition only, (c) both.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from analyze_condition_net import FEATURES, OUT_DIR, forward, load_run
from extract_statistical_moments import compute_window_moments

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
SEQ_LEN, RUL_CAP = 30, 125


def build_windows():
    cfg = json.load(open(os.path.join(_CODE_DIR, "datasets_kepin_config.json")))[1]
    df = pd.read_csv(os.path.join(_CODE_DIR, cfg["train_path"]), sep=r"\s+", header=None)
    df = df.iloc[:, :len(cfg["column_names"])]
    df.columns = cfg["column_names"]
    feats = cfg["feature_cols"]
    # 6 operating conditions are exact clusters of the rounded settings
    key = df["setting1"].round(0).astype(str) + "_" + df["setting2"].round(2).astype(str) + "_" + df["setting3"].round(0).astype(str)
    cond = pd.factorize(key)[0]
    assert cond.max() == 5, f"expected 6 operating conditions, got {cond.max() + 1}"
    df[feats] = StandardScaler().fit_transform(df[feats])

    A, comp, rul, unit = [], [], [], []
    for u in sorted(df["id"].unique()):
        m = (df["id"] == u).values
        udf = df[m].sort_values("cycle")
        X, c = udf[feats].values, cond[m][np.argsort(udf["cycle"].values)]
        n = len(X)
        for i in range(n - SEQ_LEN + 1):
            A.append(compute_window_moments(X[i:i + SEQ_LEN]))
            comp.append(np.bincount(c[i:i + SEQ_LEN], minlength=6) / SEQ_LEN)
            rul.append(min(n - (i + SEQ_LEN), RUL_CAP))
            unit.append(u)
    return np.array(A), np.array(comp), np.array(rul, float), np.array(unit)


def oof_r2(X, y, groups):
    rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=0)
    pred = cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(5))
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    data = np.load(os.path.join(_SCRIPT_DIR, "fd002_signals.npz"))
    A_saved, Y_saved = data["A_train"], data["Y_train"].ravel()
    A_raw, comp, rul, unit = build_windows()
    A_std = StandardScaler().fit_transform(A_raw)
    check = {
        "n_windows": int(len(A_raw)),
        "max_abs_diff_vs_saved_A": float(np.abs(A_std - A_saved).max()),
        "rul_matches_saved_Y": bool(np.allclose(rul, Y_saved)),
    }
    print(json.dumps(check, indent=2))

    rng = np.random.default_rng(0)
    idx = rng.choice(len(A_std), 15000, replace=False)
    X_rul, X_comp = rul[idx, None], comp[idx]
    X_both = np.hstack([X_comp, X_rul])
    g = unit[idx]

    targets = {f"signalA_{f}": A_std[idx, i] for i, f in enumerate(FEATURES)}
    for run in range(3):
        layers, s = load_run(run)
        ds, sig, _ = forward(layers, s, A_saved[idx].astype(np.float64))
        ds_c = ds - ds.mean(0)
        pc1 = ds_c @ np.linalg.svd(ds_c, full_matrices=False)[2][0]
        targets[f"run{run}_delta_s_PC1"] = pc1
        targets[f"run{run}_mean_sigma"] = sig.mean(1)

    res = {}
    for name, y in targets.items():
        res[name] = {"R2_RUL_only": oof_r2(X_rul, y, g),
                     "R2_conditions_only": oof_r2(X_comp, y, g),
                     "R2_both": oof_r2(X_both, y, g)}
        print(f"{name:24s} " + "  ".join(f"{k}={v:.3f}" for k, v in res[name].items()), flush=True)

    with open(os.path.join(OUT_DIR, "confound_operating_conditions.json"), "w") as f:
        json.dump({"sanity": check, "out_of_fold_R2": res}, f, indent=2)
    plot_r2(res)


def plot_r2(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(res)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, [res[n]["R2_conditions_only"] for n in names], 0.4, label="operating-condition mix")
    ax.bar(x + 0.2, [max(res[n]["R2_RUL_only"], 0) for n in names], 0.4, label="RUL (engine health)")
    ax.set_xticks(x, [n.replace("_", "\n", 1) for n in names], fontsize=8)
    ax.set_ylabel("out-of-fold R²")
    ax.set_ylim(0, 1.15)
    ax.set_title("How much of each quantity can be explained by...")
    ax.legend(loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confound_r2.png"), dpi=150)


if __name__ == "__main__":
    main()
