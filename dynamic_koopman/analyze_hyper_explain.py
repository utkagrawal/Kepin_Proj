#!/usr/bin/env python3
"""
Explainability analysis of the health-driven hypernetwork Koopman (arm A: eig_hyper_hh_mono), any C-MAPSS dataset.

In arm A the hypernetwork reads ONLY the health index h, so every change of the Koopman rule is caused by h.
For each run (0..2) of the chosen protocol:
  1. knockout on the test engines: normal | same K for all | h shuffled | spectral features fixed
  2. over full training-engine lives: h, mean decay rho, number of slow modes (rho > 0.3), binned by flights
     before failure; per-engine "does h fall over life" (Spearman)
  3. run 0 only:
     - eigenvalues of one engine's generated rule when healthy vs near failure (unit-circle plot)
     - the "health dial": override h = 0..1 for a few test engines and record predicted RUL and slow modes
       (a what-if explanation: how the prediction responds to health through the Koopman rule alone)
     - per-mode rho vs h (heatmap)
Usage: python analyze_hyper_explain.py FD002 [clean|paper]
Output: reports/hyper_explain/<ds>_<protocol>.npz + .json
"""
import json, os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core")); sys.path.insert(0, _DIR)
import numpy as np, tensorflow as tf
from scipy.stats import spearmanr
import kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

DS = sys.argv[1]
PROTO = sys.argv[2] if len(sys.argv) > 2 else "clean"
ARM = "eig_hyper_hh_mono"
OUT = os.path.join(_DIR, "reports", "hyper_explain"); os.makedirs(OUT, exist_ok=True)
D = load_cmapss(DS)
Xte, yte = D["X_test"].astype(np.float32), D["Y_test"].ravel()
Ute = np.concatenate([D["hist_test"], D["age_test"][:, None] / 100.0], 1).astype(np.float32)
units = D["units_train"]
step = 4 if len(np.unique(units)) > 150 else 2
keep = np.isin(units, np.unique(units)[::step])
Xtr = D["X_train"][keep].astype(np.float32)
Utr = np.concatenate([D["hist_train"], D["age_train"][:, None] / 100.0], 1)[keep].astype(np.float32)
u_tr = units[keep]
rem = np.zeros(len(u_tr), int)
for u in np.unique(u_tr):
    i = np.where(u_tr == u)[0]; rem[i] = np.arange(len(i))[::-1]
seq_len, n_feat = Xte.shape[1], Xte.shape[2]
cfg = CMAPSS_CONFIGS[f"CMAPSS_{DS}"]["arch_override"].copy()
cfg["kernels"] = [k if k % 2 else k - 1 for k in [min(k, seq_len) for k in cfg["kernels"]]]
orig_spec = kepin_model.extract_spectral_features
BINS = np.arange(0, 310, 10)          # flights before failure, binned


def run(m, X, U, batch=1024):
    p, lr, th, h = [], [], [], []
    for i in range(0, len(X), batch):
        r, ko = m((tf.constant(X[i:i + batch]), tf.constant(U[i:i + batch])), training=False)
        p.append(r.numpy().ravel()); lr.append(ko["adapt_log_rho"].numpy()); th.append(ko["adapt_theta"].numpy())
        h.append(ko["health"].numpy().ravel())
    return np.concatenate(p), np.concatenate(lr), np.concatenate(th), np.concatenate(h)


def binned(v):
    b = np.minimum(rem, 300) // 10 * 10
    return np.array([v[b == x].mean() if (b == x).any() else np.nan for x in BINS])


rmse = lambda p: float(np.sqrt(np.mean((p - yte) ** 2)))
res, curves = {"dataset": DS, "protocol": PROTO, "runs": []}, {}
for r in range(3):
    tf.keras.backend.clear_session(); kepin_model.extract_spectral_features = orig_spec
    m = DynamicKePIN((seq_len, n_feat), arch_config=cfg, koopman="eig", koopman_mode="hyper", hist_health=True)
    m((tf.constant(Xte[:2]), tf.constant(Ute[:2])), training=False)
    m.load_weights(f"{_DIR}/runs/{PROTO}/{DS}/{ARM}/run{r}.weights.h5")
    K = m.koopman
    p, lr, th, h = run(m, Xte, Ute)
    k = {"normal": rmse(p)}
    lm = tf.constant(lr.mean(0, keepdims=True)); tm = tf.constant(th.mean(0, keepdims=True))
    orig = K.hyper_params
    K.hyper_params = lambda z, signal=None: (tf.tile(lm, [tf.shape(z)[0], 1]), tf.tile(tm, [tf.shape(z)[0], 1]), None)
    k["same K for all engines"] = rmse(run(m, Xte, Ute)[0]); K.hyper_params = orig
    Us = Ute.copy(); Us[:, :42] = Ute[np.random.default_rng(r).permutation(len(Ute)), :42]
    k["h shuffled"] = rmse(run(m, Xte, Us)[0])
    _, ko = m((tf.constant(Xte), tf.constant(Ute)), training=False)
    sm = tf.reduce_mean(tf.cast(orig_spec(ko["eigenvalues"], top_k=m.spectral_k), tf.float32), 0)
    kepin_model.extract_spectral_features = lambda eig, top_k=4: sm
    k["spectral features fixed"] = rmse(run(m, Xte, Ute)[0]); kepin_model.extract_spectral_features = orig_spec

    _, lr_tr, th_tr, h_tr = run(m, Xtr, Utr)
    rho = np.exp(lr_tr); slow = (rho > 0.3).sum(1)
    falls = [spearmanr(np.arange(len(i)), h_tr[i])[0] for i in (np.where(u_tr == u)[0] for u in np.unique(u_tr))]
    sgn = np.sign(np.corrcoef(h_tr, np.minimum(rem, 125))[0, 1])     # +1 if h is "health", -1 if flipped
    healthy, near = rem >= 150, rem < 30
    stats = {"knockout": k,
             "h_corr_rul_capped": float(np.corrcoef(h_tr, np.minimum(rem, 125))[0, 1]),
             "h_falls_over_life_share": float(np.mean(np.array(falls) < 0)) if sgn > 0 else float(np.mean(np.array(falls) > 0)),
             "corr_mean_rho_h": float(np.corrcoef(rho.mean(1), h_tr)[0, 1]),
             "corr_mean_rho_rul": float(np.corrcoef(rho.mean(1), np.minimum(rem, 125))[0, 1]),
             "mean_rho_healthy": float(rho[healthy].mean()), "mean_rho_near_failure": float(rho[near].mean()),
             "slow_modes_healthy": float(slow[healthy].mean()), "slow_modes_near_failure": float(slow[near].mean()),
             "median_half_life_healthy": float(np.median(np.log(0.5) / np.minimum(lr_tr[healthy], -1e-6))),
             "median_half_life_near_failure": float(np.median(np.log(0.5) / np.minimum(lr_tr[near], -1e-6))),
             "h_flipped": bool(sgn < 0)}
    res["runs"].append(stats)
    curves[f"r{r}_h"], curves[f"r{r}_rho"], curves[f"r{r}_slow"] = binned(h_tr), binned(rho.mean(1)), binned(slow)
    print(DS, PROTO, r, json.dumps({a: round(b, 2) for a, b in k.items()}),
          {a: round(b, 3) if isinstance(b, float) else b for a, b in stats.items() if a != "knockout"}, flush=True)

    if r == 0:
        # per-engine h trajectories (a few engines) and eigenvalues healthy vs near failure
        eng = [u for u in np.unique(u_tr) if (u_tr == u).sum() >= 180][:6]
        for j, u in enumerate(eng):
            i = np.where(u_tr == u)[0]
            curves[f"traj{j}_rem"], curves[f"traj{j}_h"] = rem[i], h_tr[i]
            curves[f"traj{j}_slow"] = slow[i]
        i = np.where(u_tr == eng[0])[0]
        early, late = i[np.argmax(rem[i] <= 200)] if (rem[i] >= 200).any() else i[0], i[-1]
        curves["eig_early"] = rho[early] * np.exp(1j * th_tr[early]); curves["eig_late"] = rho[late] * np.exp(1j * th_tr[late])
        curves["eig_early_rem"], curves["eig_late_rem"] = rem[early], rem[late]
        # per-mode rho vs h
        hb = np.linspace(0, 1, 11)
        curves["mode_rho_by_h"] = np.array([np.median(rho[(h_tr >= hb[q]) & (h_tr < hb[q + 1])], 0)
                                            if ((h_tr >= hb[q]) & (h_tr < hb[q + 1])).sum() > 20
                                            else np.full(rho.shape[1], np.nan) for q in range(10)])
        # health dial: override h for 5 test engines spread over true RUL
        order = np.argsort(yte); pick = order[np.linspace(5, len(order) - 6, 5).astype(int)]
        dial = np.linspace(0, 1, 21)
        rul_d, slow_d = [], []
        hh = m.history_health
        for hv in dial:
            m.history_health = lambda c, hv=hv: tf.fill([tf.shape(c)[0], 1], tf.constant(hv, tf.float32))
            pd_, lrd, _, _ = run(m, Xte[pick], Ute[pick])
            rul_d.append(pd_); slow_d.append((np.exp(lrd) > 0.3).sum(1))
        m.history_health = hh
        curves["dial_h"], curves["dial_rul"], curves["dial_slow"] = dial, np.array(rul_d), np.array(slow_d)
        curves["dial_true"], curves["dial_hreal"] = yte[pick], h[pick]

curves["bins"] = BINS
np.savez(os.path.join(OUT, f"{DS}_{PROTO}.npz"), **curves)
json.dump(res, open(os.path.join(OUT, f"{DS}_{PROTO}.json"), "w"), indent=2)
print("saved", DS, PROTO)
