"""Is the latent trajectory straightened by the encoder? (FD002, paper runs)

For each trained model, compute the latent of every training window (cond_norm -> EMA -> encoder),
take the window's last state z_T, and project it on the direction that best predicts RUL
(ridge, out-of-fold by engine) -> a 1-D "latent health" per window. Per engine, compare the
slope of this latent health over the last 50 flights with the slope 100-150 flights before failure.
Ratio ~ 1: the encoder made the degradation path straight (linear in time).
Ratio >> 1: the latent still accelerates near failure (nonlinearity left for the dynamics).
Reference: the condition-normalised sensor index accelerates ~6x (median late / mid slope).
"""
import os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core")); sys.path.insert(0, _DIR)
import numpy as np, tensorflow as tf
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

D = load_cmapss("FD002")
X, U, units, Y = D["X_train_raw"], D["U_train"][..., :6], D["units_train"], D["Y_train"].ravel()
Uf = U.reshape(len(X), -1)
a = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
a["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in a["kernels"]]]
dummy = (np.zeros((6, 24), np.float32), np.ones((6, 24), np.float32))
MODELS = {
    "KePIN head (svd_static_cn): K ignored by the head": ("svd_static_cn", dict(koopman="svd", koopman_mode="static")),
    "wear model (eig_static_wear_cn): RUL only through linear K": ("eig_static_wear_cn",
                                                                    dict(koopman="eig", koopman_mode="static", head="wear")),
    "self-supervised encoder (no RUL labels)": ("SSL", dict(koopman="eig", koopman_mode="static", head="wear")),
}
ONLY = sys.argv[1:]          # optional: names (prefixes) of the models to run
# true remaining flights (uncapped) per window, for the slope comparison
rem = np.zeros(len(units), int)
for u in np.unique(units):
    idx = np.where(units == u)[0]; rem[idx] = np.arange(len(idx))[::-1]

def slopes(score):
    late, mid = [], []
    for u in np.unique(units):
        i = units == u; r, s = rem[i], score[i]
        A, B = r < 50, (r >= 100) & (r < 150)
        if A.sum() > 20 and B.sum() > 20:
            late.append(np.polyfit(-r[A], s[A], 1)[0]); mid.append(np.polyfit(-r[B], s[B], 1)[0])
    late, mid = np.array(late), np.array(mid)
    return np.median(late) / np.median(mid), np.mean(late > mid), len(late)

for name, (arm, kw) in MODELS.items():
    if ONLY and not any(name.startswith(o) for o in ONLY):
        continue
    for run in range(3):
        m = DynamicKePIN((30, 24), arch_config=a, exo_dim=6, cond_norm=True, cond_stats=dummy,
                         ema_alpha=D["ema_alpha"], **kw)
        m((tf.constant(X[:2]), tf.constant(Uf[:2])), training=False)
        if arm == "SSL":
            m.load_frozen_encoder(f"{_DIR}/runs/ssl/FD002/run{run}_encoder.npz")
        else:
            m.load_weights(f"{_DIR}/runs/paper/FD002/{arm}/run{run}.weights.h5")
        Z = np.concatenate([m._condition_aware_latent((tf.constant(X[i:i + 2048]), tf.constant(Uf[i:i + 2048])),
                                                      False)[0][:, -1].numpy() for i in range(0, len(X), 2048)])
        # latent health = out-of-fold linear projection that best predicts the UNCAPPED remaining flights
        p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Z, -rem.astype(float), groups=units, cv=GroupKFold(5))
        ratio, share, n = slopes(p)
        r2 = 1 - np.sum((p + rem) ** 2) / np.sum((rem - rem.mean()) ** 2)
        print(f"{name[:60]:60s} run{run}: late/mid slope ratio {ratio:4.1f}x, faster late in {100*share:3.0f}% of {n} engines "
              f"(linear R2 of remaining flights {r2:.2f})", flush=True)
        tf.keras.backend.clear_session()
