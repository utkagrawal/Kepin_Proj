"""Knockout in the condition-aware wear model: force one operator for all engines at test time."""
import sys, os
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core")); sys.path.insert(0, _DIR)
import numpy as np, tensorflow as tf
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN
D = load_cmapss("FD002")
X, y = D["X_test_raw"], D["Y_test"].ravel()
U = D["U_test"][..., :6].reshape(len(X), -1)
a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
ARMS = {"gate": ("eig_health_wear_cn_ord_gap0_mono", dict(health_map="ordered", gap_init=0.0)),
        "mixture": ("eig_health_wear_cn_mix_mono", dict(adapt="mixture")),
        "local K": ("eig_health_wear_cn_eng", dict(adapt="engine"))}
def rmse(m):
    return float(np.sqrt(np.mean((m((tf.constant(X), tf.constant(U)), training=False)[0].numpy().ravel() - y) ** 2)))
for name, (arm, kw) in ARMS.items():
    rows = []
    for run in range(3):
        dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))
        m = DynamicKePIN((30, 24), arch_config=a, koopman="eig", koopman_mode="health", exo_dim=6, head="wear",
                         cond_norm=True, cond_stats=dummy, ema_alpha=D["ema_alpha"], **kw)
        m((tf.constant(X[:2]), tf.constant(U[:2])), training=False)
        m.load_weights(f"{_DIR}/runs/paper/FD002/{arm}/run{run}.weights.h5")
        K = m.koopman
        r = {"normal": rmse(m)}
        if name == "gate":
            o = K.health
            K.health = lambda z: tf.ones([tf.shape(z)[0], 1]); r["all healthy operator"] = rmse(m)
            K.health = lambda z: tf.zeros([tf.shape(z)[0], 1]); r["all failing operator"] = rmse(m)
            K.health = o
        elif name == "mixture":
            o = K.regime_probs
            for i in range(3):
                K.regime_probs = lambda z, i=i: tf.one_hot(tf.fill([tf.shape(z)[0]], i), 3); r[f"all regime {i}"] = rmse(m)
            K.regime_probs = o
        else:
            o = K.engine_params
            K.engine_params = lambda zt: (tf.ones([tf.shape(zt)[0], 1]) * tf.math.log_sigmoid(K.s),
                                          tf.ones([tf.shape(zt)[0], 1]) * K.theta, None)
            r["fleet operator for all"] = rmse(m); K.engine_params = o
        # remove the Koopman transient entirely: health = H0-free readout? -> zero the fading modes
        o = K.health_rollout
        def no_transient(z_last, horizon, w, b, lr=None, th=None, o=o):
            h = o(z_last, horizon, w, b, lr, th)
            return tf.broadcast_to(h[:, :1], tf.shape(h))          # keep current health, drop fading part
        K.health_rollout = no_transient; r["fading modes frozen at k=0"] = rmse(m); K.health_rollout = o
        rows.append(r); tf.keras.backend.clear_session()
    print(f"\n{name}")
    for k in rows[0]:
        v = [r[k] for r in rows]
        print(f"  {k:28s} " + "  ".join(f"{x:6.2f}" for x in v) + f"   | change {np.mean(v) - np.mean([r['normal'] for r in rows]):+.2f}")
