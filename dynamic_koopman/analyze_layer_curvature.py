#!/usr/bin/env python3
"""
Where inside the encoder is the degradation curve flattened? (FD002, run 0)

The condition-normalised, smoothed sensors (the encoder's input) speed up ~5x near failure;
the self-supervised latent keeps ~2.8x and the RUL-trained latent ~1.8x (same measure, see
analyze_latent_curvature.py). Here the encoder is stepped through block by block and the same
measure is taken after every stage, on the LAST time step of each training window:
  curvature = median slope over the last 50 flights / median slope 100-150 flights before failure
of the out-of-fold ridge projection that best predicts the remaining flights (folds = engines).
Also the linear R^2 of the remaining flights from that stage.
Output: reports/latent_curvature/layers.json
"""
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

D = load_cmapss("FD002")
X, U, units = D["X_train_raw"].astype(np.float32), D["U_train"][..., :6], D["units_train"]
Uf = U.reshape(len(X), -1).astype(np.float32)
rem = np.zeros(len(units), int)
for u in np.unique(units):
    i = np.where(units == u)[0]
    rem[i] = np.arange(len(i))[::-1]
a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))


def curvature(F):
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), F, -rem.astype(float), groups=units,
                          cv=GroupKFold(5))
    late, mid = [], []
    for u in np.unique(units):
        i = units == u
        r, s = rem[i], p[i]
        A, B = r < 50, (r >= 100) & (r < 150)
        if A.sum() > 20 and B.sum() > 20:
            late.append(np.polyfit(-r[A], s[A], 1)[0])
            mid.append(np.polyfit(-r[B], s[B], 1)[0])
    late, mid = np.array(late), np.array(mid)
    r2 = 1 - np.sum((p + rem) ** 2) / np.sum((rem - rem.mean()) ** 2)
    return float(np.median(late) / np.median(mid)), float(np.mean(late > mid)), float(r2)


def stages(m, x, u):
    """Replicates _condition_aware_latent + encode(), returning the last time step after each stage."""
    out = {}
    u3 = tf.reshape(u, [-1, x.shape[1], 6])
    x = x * tf.einsum("btc,cf->btf", u3, m.cn_gamma) + tf.einsum("btc,cf->btf", u3, m.cn_beta)
    x = tf.einsum("ts,bsf->btf", m.ema_matrix, x)
    out["0 input (cond_norm + EMA)"] = x
    x = tf.nn.relu(m.input_bn(m.input_proj(x), training=False))
    out["1 input_proj + BN + ReLU"] = x
    for i, block in enumerate(m.encoder_blocks):
        residual = x
        h = tf.nn.relu(block["bn1"](block["conv1"](x), training=False))
        h = block["bn2"](block["conv2"](h), training=False)
        se = block["se_reshape"](block["se_dense2"](block["se_dense1"](block["se_gap"](h))))
        h = block["se_mul"]([h, se])
        if block["skip_proj"] is not None:
            residual = block["skip_bn"](block["skip_proj"](residual), training=False)
        x = tf.nn.relu(h + residual)
        out[f"{2 + i} conv block {i}"] = x
    k = 2 + len(m.encoder_blocks)
    x = m.bilstm(x, training=False)
    out[f"{k} BiLSTM"] = x
    x = m.lstm_ln(x, training=False)
    out[f"{k + 1} LayerNorm after LSTM"] = x
    x = m.post_lstm_bn(m.post_lstm_proj(x), training=False)
    out[f"{k + 2} post-LSTM proj + BN"] = x
    x = m.mha_ln(x + m.mha(x, x, training=False), training=False)
    out[f"{k + 3} attention + LayerNorm"] = x
    x = m.ff_ln(x + m.ff_dense2(m.ff_dense1(x)), training=False)
    out[f"{k + 4} feed-forward + LayerNorm"] = x
    x = m.latent_proj(x)
    out[f"{k + 5} latent projection"] = x
    x = m.latent_bn(x, training=False)
    out[f"{k + 6} latent BN (= latent z)"] = x
    return {n: v[:, -1].numpy() for n, v in out.items()}


ENCODERS = {"self-supervised (no RUL)": lambda m: m.load_frozen_encoder(f"{_DIR}/runs/ssl/FD002/run0_encoder.npz"),
            "RUL-trained (wear model)": lambda m: m.load_weights(f"{_DIR}/runs/paper/FD002/eig_static_wear_cn/run0.weights.h5")}
res = {}
for name, load in ENCODERS.items():
    tf.keras.backend.clear_session()
    m = DynamicKePIN((30, 24), arch_config=a, koopman="eig", koopman_mode="static", head="wear", exo_dim=6,
                     cond_norm=True, cond_stats=dummy, ema_alpha=D["ema_alpha"])
    m((tf.constant(X[:2]), tf.constant(Uf[:2])), training=False)
    load(m)
    acts = {}
    for i in range(0, len(X), 2048):
        for n, v in stages(m, tf.constant(X[i:i + 2048]), tf.constant(Uf[i:i + 2048])).items():
            acts.setdefault(n, []).append(v)
    print(f"\n{name}")
    res[name] = {}
    for n, v in acts.items():
        c, share, r2 = curvature(np.concatenate(v))
        res[name][n] = {"curvature": c, "share_faster_late": share, "linear_r2_remaining": r2}
        print(f"  {n:34s} curvature {c:4.1f}x  faster late in {100 * share:3.0f}%  linear R2 {r2:.2f}", flush=True)
os.makedirs(os.path.join(_DIR, "reports", "latent_curvature"), exist_ok=True)
json.dump(res, open(os.path.join(_DIR, "reports", "latent_curvature", "layers.json"), "w"), indent=2)
