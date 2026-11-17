"""Does the RUL head use the adaptive operator? Switch parts off at test time (no retraining)."""
import sys, os
DK = "/userhome/mtech/a.utkarsh/RCKePIN/dynamic_koopman"
sys.path.insert(0, os.path.join(DK, "kepin_core")); sys.path.insert(0, DK)
import numpy as np, tensorflow as tf
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN
D = load_cmapss("FD002")
X, U, y = D["X_test_raw"], D["U_test"][..., :6].reshape(len(D["X_test_raw"]), -1), D["Y_test"].ravel()
a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
ARMS = {"gate_hs10": dict(health_map="ordered", gap_init=0.0, head_state=10),
        "gate_fc": dict(health_map="ordered", gap_init=0.0, head_state=[1, 5, 10, 25, 50]),
        "mix_fc": dict(adapt="mixture", head_state=[1, 5, 10, 25, 50]),
        "eng_fc": dict(adapt="engine", head_state=[1, 5, 10, 25, 50])}
DIRS = {"gate_hs10": "eig_health_cn_ctrl_ord_gap0_hs10_mono", "gate_fc": "eig_health_cn_ctrl_ord_gap0_fc_mono",
        "mix_fc": "eig_health_cn_ctrl_mix_fc_mono", "eng_fc": "eig_health_cn_ctrl_eng_fc"}
def rmse(m): return float(np.sqrt(np.mean((m((tf.constant(X), tf.constant(U)), training=False)[0].numpy().ravel() - y) ** 2)))
for arm, kw in ARMS.items():
    rows = []
    for run in range(3):
        dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))
        m = DynamicKePIN((30, 24), arch_config=a, koopman="eig", koopman_mode="health", exo_dim=6, cond_norm=True,
                         koop_control=True, cond_stats=dummy, ema_alpha=D["ema_alpha"], **kw)
        m((tf.constant(X[:2]), tf.constant(U[:2])), training=False)
        m.load_weights(f"{DK}/runs/paper/FD002/{DIRS[arm]}/run{run}.weights.h5")
        K = m.koopman
        r = {"normal": rmse(m)}
        if kw.get("adapt", "gate") == "gate":
            orig = K.health
            K.health = lambda z: tf.ones([tf.shape(z)[0], 1]); r["all healthy (h=1)"] = rmse(m)
            K.health = lambda z: tf.zeros([tf.shape(z)[0], 1]); r["all failing (h=0)"] = rmse(m)
            K.health = orig
        elif kw["adapt"] == "mixture":
            orig = K.regime_probs
            for i, name in enumerate(["all regime 0", "all regime 1", "all regime 2"]):
                K.regime_probs = lambda z, i=i: tf.one_hot(tf.fill([tf.shape(z)[0]], i), 3); r[name] = rmse(m)
            K.regime_probs = orig
        else:
            orig = K.engine_params
            K.engine_params = lambda zt: (tf.ones([tf.shape(zt)[0], 1]) * tf.math.log_sigmoid(K.s),
                                          tf.ones([tf.shape(zt)[0], 1]) * K.theta, None)
            r["fleet K for all (no per-engine)"] = rmse(m); K.engine_params = orig
        origp = K.propagate
        K.propagate = lambda z, lr, th, k: tf.zeros([tf.shape(z)[0], 2 * K.n_modes]); r["forecast states zeroed"] = rmse(m)
        K.propagate = origp
        rows.append(r); tf.keras.backend.clear_session()
    print(f"\n{arm}")
    for k in rows[0]:
        v = [r[k] for r in rows]; print(f"  {k:32s} " + "  ".join(f"{x:6.2f}" for x in v) + f"   | change vs normal {np.mean(v) - np.mean([r['normal'] for r in rows]):+.2f}")
