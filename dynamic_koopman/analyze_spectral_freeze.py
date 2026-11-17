#!/usr/bin/env python3
"""
Feature replacement experiment: does KePIN's predictor actually USE the 12 Koopman
spectral features?

Models (FD002, clean protocol, runs 0-2; trained with train.py):
  eig_static  fixed Koopman rule (control: its features are already identical for every engine)
  eig_exo     dynamic version A: the rule changes with the operating condition at every flight
  eig_health  dynamic version B: the rule's decay is shifted by a learned per-window health value
In A and B the 12 features differ from engine to engine.

What is replaced, and by what:
  For each trained model we compute its 12 spectral features for all 253 FD002 test engines,
  then replace every engine's features by the AVERAGE over those 253 engines (one 12-number
  vector, computed from the same model). The average is used, not zeros or random numbers,
  so the predictor still receives typical, in-range values: only the engine-to-engine
  differences are removed. If the predictor relied on those differences, the error would rise.

How: KePINModel.call obtains the features via kepin_model.extract_spectral_features; the
function is temporarily swapped for one that returns the constant vector (see `frozen`).

Output: reports/spectral_freeze/summary.json
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf

import kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

RUNS = os.path.join(_DIR, "runs", "clean", "FD002")
OUT = os.path.join(_DIR, "reports", "spectral_freeze")
ARMS = [("eig_static", "static", "fixed rule (control)"),
        ("eig_exo", "exo", "dynamic A: operating condition"),
        ("eig_health", "health", "dynamic B: learned health")]


def main():
    os.makedirs(OUT, exist_ok=True)
    D = load_cmapss("FD002")
    X, Y = D["X_test"], D["Y_test"].ravel()
    U = D["U_test"].reshape(len(X), -1)
    arch = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
    arch["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in arch["kernels"]]]
    original = kepin_model.extract_spectral_features
    rmse = lambda p: float(np.sqrt(np.mean((p - Y) ** 2)))
    results = []

    for arm, mode, label in ARMS:
        for run in range(3):
            exo = mode == "exo"
            m = DynamicKePIN((30, X.shape[2]), arch_config=arch, koopman="eig", koopman_mode=mode,
                             exo_dim=6 if exo else 0)
            inp = (X, U) if exo else X
            m(tuple(tf.constant(v[:2]) for v in inp) if exo else tf.constant(X[:2]), training=False)
            m.load_weights(os.path.join(RUNS, arm, f"run{run}.weights.h5"))

            # 1) normal prediction, and the 12 features of every test engine
            kepin_model.extract_spectral_features = original
            pred, ko = m(inp, training=False)
            pred = pred.numpy().ravel()
            feats = original(ko["eigenvalues"], top_k=m.spectral_k).numpy()   # (253, 12) or (12,)
            const = feats.mean(axis=0) if feats.ndim > 1 else feats            # average over engines

            # 2) same model, every engine gets the SAME (average) 12 features
            def frozen(eigenvalues, top_k=5, c=const):
                c = tf.constant(c[None], tf.float32)
                return tf.tile(c, [tf.shape(eigenvalues)[0], 1]) if len(eigenvalues.shape) > 1 else c[0]
            kepin_model.extract_spectral_features = frozen
            pred_c, _ = m(inp, training=False)
            pred_c = pred_c.numpy().ravel()
            kepin_model.extract_spectral_features = original

            spread = float(feats.std(axis=0).mean()) if feats.ndim > 1 else 0.0
            r = {"arm": arm, "label": label, "run": run, "rmse_real_features": rmse(pred),
                 "rmse_constant_features": rmse(pred_c),
                 "max_prediction_change_flights": float(np.abs(pred - pred_c).max()),
                 "mean_feature_spread_across_engines": spread,
                 "constant_used": [float(v) for v in const]}
            results.append(r)
            print(f"{label:32s} run{run}: RMSE {r['rmse_real_features']:.3f} -> {r['rmse_constant_features']:.3f} "
                  f"| max change {r['max_prediction_change_flights']:.3f} flights "
                  f"| features vary across engines by {spread:.4f} (avg std)", flush=True)
            tf.keras.backend.clear_session()
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
