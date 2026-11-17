"""Knockout for the hypernetwork Koopman in original KePIN (FD002 paper runs, test engines).

normal              : as trained
same K for all      : every engine gets the AVERAGE generated operator (mean log_rho / theta / gate over
                      the test engines) -> removes the per-engine adaptation, keeps the operator's level
age shuffled        : (age arms) the age input is permuted across engines -> is the age information used?
spectral fixed      : the head's spectral features are replaced by their mean -> is the per-engine K used by
                      the head through the spectral channel (the only path from K to the RUL in KePIN)?
"""
import os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core")); sys.path.insert(0, _DIR)
import numpy as np, tensorflow as tf
import kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

D = load_cmapss("FD002")
X, y, age = D["X_test"], D["Y_test"].ravel(), D["age_test"][:, None] / 100.0
cfg = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
cfg["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in cfg["kernels"]]]
ARMS = {"hyper (window)": ("eig_hyper", dict()), "hyper + age": ("eig_hyper_age", dict(hyper_ctx_dim=1)),
        "hyper + age + rank": ("eig_hyper_age_rank", dict(hyper_ctx_dim=1, hyper_rank=True))}
orig_spec = kepin_model.extract_spectral_features

def run_model(m, ctx):
    inp = (tf.constant(X), tf.constant(ctx.astype(np.float32))) if ctx is not None else tf.constant(X)
    return m(inp, training=False)

def rmse(m, ctx):
    return float(np.sqrt(np.mean((run_model(m, ctx)[0].numpy().ravel() - y) ** 2)))

for name, (arm, kw) in ARMS.items():
    rows = []
    for run in range(3):
        tf.keras.backend.clear_session()
        kepin_model.extract_spectral_features = orig_spec
        m = DynamicKePIN((30, 24), arch_config=cfg, koopman="eig", koopman_mode="hyper", exo_dim=6 if kw else 0, **kw)
        ctx = age if kw else None
        run_model(m, None if ctx is None else ctx[:2]) if False else None
        m((tf.constant(X[:2]), tf.constant(ctx[:2].astype(np.float32))) if ctx is not None else tf.constant(X[:2]), training=False)
        m.load_weights(f"{_DIR}/runs/paper/FD002/{arm}/run{run}.weights.h5")
        K = m.koopman
        r = {"normal": rmse(m, ctx)}
        # average operator for everyone
        _, ko = run_model(m, ctx)
        lr_mean = tf.reduce_mean(ko["adapt_log_rho"], 0, keepdims=True)
        th_mean = tf.reduce_mean(ko["adapt_theta"], 0, keepdims=True)
        orig = K.hyper_params
        K.hyper_params = lambda z, signal=None: (tf.tile(lr_mean, [tf.shape(z)[0], 1]),
                                                 tf.tile(th_mean, [tf.shape(z)[0], 1]), None)
        r["same K for all engines"] = rmse(m, ctx)
        K.hyper_params = orig
        if ctx is not None:
            r["age shuffled"] = rmse(m, ctx[np.random.default_rng(run).permutation(len(ctx))])
        # spectral features fixed at their mean
        spec_mean = tf.reduce_mean(tf.cast(orig_spec(ko["eigenvalues"], top_k=m.spectral_k), tf.float32), 0)
        kepin_model.extract_spectral_features = lambda eig, top_k=4: spec_mean
        r["spectral features fixed"] = rmse(m, ctx)
        kepin_model.extract_spectral_features = orig_spec
        rows.append(r)
    print(f"\n{name}")
    for k in rows[0]:
        v = [rr[k] for rr in rows]
        print(f"  {k:26s} " + "  ".join(f"{x:6.2f}" for x in v) + f"   | change {np.mean(v) - np.mean([rr['normal'] for rr in rows]):+.2f}")
