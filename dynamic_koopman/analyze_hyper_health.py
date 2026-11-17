#!/usr/bin/env python3
"""
Arm A of the history-health hypernetwork (FD002, paper runs): K is generated from the health index h ONLY.

1. Knockout on the 253 test engines:
     normal | same K for all engines (mean generated operator) | h shuffled across engines |
     spectral features fixed (the only path from K to the RUL in KePIN)
2. Readable Koopman: on full training-engine lives, how the generated operator changes with h:
     - mean decay rate rho (per step) vs h and vs true remaining flights
     - per-mode half-life  (steps until a mode keeps half its size: ln 0.5 / ln rho)
     - "slow modes": modes with rho > 0.3 per step (how big the latent space stays)
Output: reports/hyper_health/armA.json + armA.png
"""
import json, os, sys
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core")); sys.path.insert(0, _DIR)
import numpy as np, tensorflow as tf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS
from dk_data import load_cmapss
from dk_model import DynamicKePIN

ARM = sys.argv[1] if len(sys.argv) > 1 else "eig_hyper_hh_mono"
PROTO = sys.argv[2] if len(sys.argv) > 2 else "paper"
D = load_cmapss("FD002")
Xte, yte = D["X_test"].astype(np.float32), D["Y_test"].ravel()
Ute = np.concatenate([D["hist_test"], D["age_test"][:, None] / 100.0], 1).astype(np.float32)
units = D["units_train"]
keep = np.isin(units, np.unique(units)[::4])          # every 4th training engine, full lives
Xtr = D["X_train"][keep].astype(np.float32)
Utr = np.concatenate([D["hist_train"], D["age_train"][:, None] / 100.0], 1)[keep].astype(np.float32)
u_tr = units[keep]
rem = np.zeros(len(u_tr), int)
for u in np.unique(u_tr):
    i = np.where(u_tr == u)[0]; rem[i] = np.arange(len(i))[::-1]
cfg = CMAPSS_CONFIGS["CMAPSS_FD002"]["arch_override"].copy()
cfg["kernels"] = [k if k % 2 else k - 1 for k in [min(k, 30) for k in cfg["kernels"]]]
orig_spec = kepin_model.extract_spectral_features


def run(m, X, U, batch=1024):
    p, lr, h = [], [], []
    for i in range(0, len(X), batch):
        r, ko = m((tf.constant(X[i:i + batch]), tf.constant(U[i:i + batch])), training=False)
        p.append(r.numpy().ravel()); lr.append(ko["adapt_log_rho"].numpy()); h.append(ko["health"].numpy().ravel())
    return np.concatenate(p), np.concatenate(lr), np.concatenate(h)


rmse = lambda p: float(np.sqrt(np.mean((p - yte) ** 2)))
knock, life = [], []
for r in range(3):
    tf.keras.backend.clear_session(); kepin_model.extract_spectral_features = orig_spec
    m = DynamicKePIN((30, 24), arch_config=cfg, koopman="eig", koopman_mode="hyper", hist_health=True)
    m((tf.constant(Xte[:2]), tf.constant(Ute[:2])), training=False)
    m.load_weights(f"{_DIR}/runs/{PROTO}/FD002/{ARM}/run{r}.weights.h5")
    K = m.koopman
    p, lr, h = run(m, Xte, Ute)
    k = {"normal": rmse(p)}
    _, ko = m((tf.constant(Xte), tf.constant(Ute)), training=False)
    lm = tf.reduce_mean(ko["adapt_log_rho"], 0, keepdims=True); tm = tf.reduce_mean(ko["adapt_theta"], 0, keepdims=True)
    orig = K.hyper_params
    K.hyper_params = lambda z, signal=None: (tf.tile(lm, [tf.shape(z)[0], 1]), tf.tile(tm, [tf.shape(z)[0], 1]), None)
    k["same K for all engines"] = rmse(run(m, Xte, Ute)[0]); K.hyper_params = orig
    Us = Ute.copy(); Us[:, :42] = Ute[np.random.default_rng(r).permutation(len(Ute)), :42]
    k["h shuffled"] = rmse(run(m, Xte, Us)[0])
    sm = tf.reduce_mean(tf.cast(orig_spec(ko["eigenvalues"], top_k=m.spectral_k), tf.float32), 0)
    kepin_model.extract_spectral_features = lambda eig, top_k=4: sm
    k["spectral features fixed"] = rmse(run(m, Xte, Ute)[0]); kepin_model.extract_spectral_features = orig_spec
    knock.append(k)
    _, lr_tr, h_tr = run(m, Xtr, Utr)
    life.append((lr_tr, h_tr))
    print(f"run {r}: " + "  ".join(f"{a} {b:.2f}" for a, b in k.items()), flush=True)

# readable operator (run 0 for plots; summary over all runs)
summ = {"knockout": {a: [kk[a] for kk in knock] for a in knock[0]}}
per_run = []
for lr_tr, h_tr in life:
    rho = np.exp(lr_tr)
    alive10 = (rho > 0.3).sum(1)
    hl = np.log(0.5) / np.minimum(lr_tr, -1e-6)
    healthy, near = rem >= 150, rem < 30
    per_run.append({"corr_mean_rho_h": float(np.corrcoef(rho.mean(1), h_tr)[0, 1]),
                    "corr_alive10_h": float(np.corrcoef(alive10, h_tr)[0, 1]),
                    "alive10_healthy": float(alive10[healthy].mean()), "alive10_near_failure": float(alive10[near].mean()),
                    "median_half_life_healthy": float(np.median(hl[healthy])),
                    "median_half_life_near_failure": float(np.median(hl[near])),
                    "modes_faster_near_failure": int((np.median(hl[near], 0) < np.median(hl[healthy], 0)).sum())})
summ["per_run"] = per_run
os.makedirs(f"{_DIR}/reports/hyper_health", exist_ok=True)
tag = f"{ARM}_{PROTO}"
json.dump(summ, open(f"{_DIR}/reports/hyper_health/{tag}.json", "w"), indent=2)
print(json.dumps(per_run, indent=1))

lr_tr, h_tr = life[0]; rho = np.exp(lr_tr); alive10 = (rho > 0.3).sum(1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
b = np.clip(rem, 0, 300) // 10 * 10
xs = np.unique(b)
ax[0].plot(-xs, [rho.mean(1)[b == v].mean() for v in xs], lw=2, color="#2a6fdb")
ax[0].set(xlabel="flights before failure", ylabel="mean decay factor rho per step", title="Generated K vs remaining life")
ax[1].plot(-xs, [alive10[b == v].mean() for v in xs], lw=2, color="#2a6fdb")
ax[1].set(xlabel="flights before failure", ylabel="modes with rho > 0.3 (slow modes)", title="Size of the latent space kept")
order = np.argsort(np.median(lr_tr[rem >= 150], 0))
hb = np.linspace(0, 1, 11)
grid = np.array([np.median(rho[(h_tr >= hb[i]) & (h_tr < hb[i + 1])][:, order], 0) if ((h_tr >= hb[i]) & (h_tr < hb[i + 1])).sum() > 20
                 else np.full(rho.shape[1], np.nan) for i in range(10)])
im = ax[2].imshow(grid.T, aspect="auto", origin="lower", extent=[0, 1, 0, rho.shape[1]], cmap="viridis")
ax[2].set(xlabel="health index h (0 = near failure, 1 = healthy)", ylabel="mode (sorted by healthy decay)", title="Per-mode rho vs h")
plt.colorbar(im, ax=ax[2])
for a in ax[:2]: a.grid(alpha=.25)
plt.tight_layout(); plt.savefig(f"{_DIR}/reports/hyper_health/{tag}.png", dpi=130)
print("saved", f"reports/hyper_health/{tag}.png")
