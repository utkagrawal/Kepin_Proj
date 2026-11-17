#!/usr/bin/env python3
"""
Self-supervised pre-training of the KePIN encoder, WITHOUT RUL labels (frozen-encoder study).

Why: an encoder trained on the (linear, capped) RUL label learns to straighten the degradation
path (sensor index accelerates ~6x near failure, RUL-trained latents only ~1.8x; see
analyze_latent_curvature.py). Then an adaptive Koopman operator has nothing left to adapt to.
Here the same encoder (cond_norm -> in-model EMA -> CNN/BiLSTM/attention -> latent) is trained
only to reconstruct its own input window from the latent sequence (denoising: the input gets the
same noise level as the main training, the target is clean). Nothing asks it to make the
degradation path linear in time.

Output: runs/ssl/<dataset>/run<k>_encoder.npz (encoder weights incl. BatchNorm statistics and
the per-condition normalisation), loaded and frozen by train.py --frozen_encoder.
Paper protocol: all training windows (no labels are used).
"""
import argparse
import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
import keras

from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

SEED = 42


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="FD002")
    p.add_argument("--run_id", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.009)   # = sqrt(0.004^2 + 0.008^2), as in training
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = os.path.join(_DIR, "runs", "ssl", args.dataset)
    os.makedirs(out, exist_ok=True)
    tf.random.set_seed(SEED + args.run_id * 100)
    np.random.seed(SEED + args.run_id * 100)

    D = load_cmapss(args.dataset)
    nc = D["n_conditions"]
    X = D["X_train_raw"].astype(np.float32)
    U = D["U_train"][..., :nc].reshape(len(X), -1).astype(np.float32)
    if args.smoke:
        X, U = X[:2000], U[:2000]
    seq_len, n_feat = X.shape[1], X.shape[2]
    # same per-condition statistics as train.py (all training time steps, std floored at 0.1)
    xs = X.reshape(-1, n_feat)
    cs = D["U_train"][..., :nc][:len(X)].reshape(-1, nc).argmax(1)
    mu = np.stack([xs[cs == c].mean(0) for c in range(nc)])
    sd = np.maximum(np.stack([xs[cs == c].std(0) for c in range(nc)]), 0.1)

    cfg = CMAPSS_CONFIGS[f"CMAPSS_{args.dataset}"]
    arch = cfg["arch_override"].copy()
    arch["kernels"] = [min(k, seq_len) for k in arch["kernels"]]
    arch["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch["kernels"]]
    model = DynamicKePIN((seq_len, n_feat), arch_config=arch, n_train=len(X), koopman="eig",
                         koopman_mode="static", exo_dim=nc, head="wear", cond_norm=True,
                         cond_stats=(mu.astype(np.float32), sd.astype(np.float32)), ema_alpha=D["ema_alpha"])
    model((tf.constant(X[:2]), tf.constant(U[:2])), training=False)
    decoder = keras.Sequential([keras.layers.Dense(64, activation="relu"), keras.layers.Dense(n_feat)],
                               name="ssl_decoder")
    decoder.build((None, seq_len, arch["latent_dim"]))
    enc_vars = [v for l in model.encoder_layers() for v in l.trainable_weights]
    train_vars = enc_vars + decoder.trainable_weights
    steps = args.epochs * int(np.ceil(len(X) / args.batch))
    opt = keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(args.lr, steps, alpha=0.01), clipnorm=1.0)

    def prepared(x, u):
        """The encoder's own input: cond_norm (fixed) -> EMA. This is the reconstruction target."""
        u3 = tf.reshape(u, [-1, seq_len, nc])
        x = x * tf.einsum("btc,cf->btf", u3, model.cn_gamma) + tf.einsum("btc,cf->btf", u3, model.cn_beta)
        return tf.einsum("ts,bsf->btf", model.ema_matrix, x)

    @tf.function
    def step(x, u):
        target = tf.stop_gradient(prepared(x, u))
        x_noisy = x + tf.random.normal(tf.shape(x), stddev=args.noise)
        with tf.GradientTape() as tape:
            z, _ = model._condition_aware_latent((x_noisy, u), True)
            loss = tf.reduce_mean(tf.square(decoder(z) - target))
        opt.apply_gradients(zip(tape.gradient(loss, train_vars), train_vars))
        return loss

    log, t0 = [], time.time()
    for ep in range(args.epochs):
        idx = np.random.permutation(len(X))
        losses = [float(step(tf.constant(X[idx[i:i + args.batch]]), tf.constant(U[idx[i:i + args.batch]])))
                  for i in range(0, len(X), args.batch)]
        log.append(float(np.mean(losses)))
        print(f"epoch {ep + 1:3d}/{args.epochs}  reconstruction MSE {log[-1]:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    # variance of the target, to read the MSE as a fraction of unexplained variance
    var = float(np.mean([np.var(prepared(tf.constant(X[i:i + 4096]), tf.constant(U[i:i + 4096])).numpy())
                         for i in range(0, len(X), 4096)]))
    W = model.encoder_weights()
    if not args.smoke:
        np.savez(os.path.join(out, f"run{args.run_id}_encoder.npz"), n=len(W), **{f"w{i}": w for i, w in enumerate(W)})
    json.dump({"dataset": args.dataset, "run_id": args.run_id, "epochs": args.epochs, "loss": log,
               "target_variance": var, "final_mse_over_variance": log[-1] / var, "time_s": time.time() - t0},
              open(os.path.join(out, f"run{args.run_id}_ssl{'_smoke' if args.smoke else ''}.json"), "w"), indent=2)
    print(f"done: final MSE {log[-1]:.5f} = {100 * log[-1] / var:.1f}% of target variance", flush=True)


if __name__ == "__main__":
    main()
