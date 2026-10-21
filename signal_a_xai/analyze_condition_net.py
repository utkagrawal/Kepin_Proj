#!/usr/bin/env python3
"""
Interpretability analysis of the Signal A condition network (FD002, runs 0-2).

The condition network is a pure function of the 4-D Signal A vector
    r = [mean, variance, skewness, kurtosis]   (standardized)
    delta_s = 2 * tanh(MLP(r))                 (128 Koopman modes)
    sigma   = sigmoid(s + delta_s)
so it can be rebuilt in NumPy from the saved weights and probed exhaustively
on all training windows without TensorFlow.

Analyses:
  1. Weight-level: first-layer input weight norms, dead ReLU units.
  2. Output geometry: saturation, effective rank (PCA) of delta_s over 128 modes.
  3. Link to degradation: correlation of the dominant delta_s direction / mean sigma with RUL.
  4. Attribution: exact gradients, Integrated Gradients and exact Shapley values
     (4 features -> 16 coalitions) of mean sigma, against a "healthy" baseline.
  5. Partial dependence of mean sigma on each moment.
  6. Seed consistency of all of the above.
"""
import itertools
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_SCRIPT_DIR, "final_signal_a_models", "FD002_Signal_A_models")
OUT_DIR = os.path.join(_SCRIPT_DIR, "results", "condition_net_interpretability_fd002")
FEATURES = ["mean", "variance", "skewness", "kurtosis"]
MAX_SHIFT = 2.0
SAT_FRAC = 0.9


def load_run(run_id):
    path = os.path.join(MODEL_DIR, f"kepin_CMAPSS_FD002_run{run_id}.weights.h5")
    with h5py.File(path, "r") as f:
        g = f["koopman/condition_net_raw/layers"]
        layers = [(g[n]["vars/0"][:], g[n]["vars/1"][:]) for n in ("dense", "dense_1", "dense_2")]
        s = f["koopman/vars/1"][:]
    return layers, s


def forward(layers, s, r):
    """Returns delta_s, sigma and cached activations for backprop."""
    (W1, b1), (W2, b2), (W3, b3) = layers
    h1p = r @ W1 + b1
    h1 = np.maximum(h1p, 0)
    h2p = h1 @ W2 + b2
    h2 = np.maximum(h2p, 0)
    raw = h2 @ W3 + b3
    t = np.tanh(raw)
    delta_s = MAX_SHIFT * t
    sigma = 1.0 / (1.0 + np.exp(-(s + delta_s)))
    return delta_s, sigma, (h1p, h2p, t)


def grad_mean_sigma(layers, s, r):
    """Exact d(mean_k sigma_k)/dr for each row of r, shape (N, 4)."""
    (W1, _), (W2, _), (W3, _) = layers
    _, sigma, (h1p, h2p, t) = forward(layers, s, r)
    d = sigma.shape[1]
    g_raw = (sigma * (1 - sigma) / d) * MAX_SHIFT * (1 - t ** 2)  # (N, d)
    g_h2 = (g_raw @ W3.T) * (h2p > 0)
    g_h1 = (g_h2 @ W2.T) * (h1p > 0)
    return g_h1 @ W1.T


def jacobian_delta_s(layers, r):
    """Exact d(delta_s)/dr, shape (N, d, 4)."""
    (W1, _), (W2, _), (W3, _) = layers
    _, _, (h1p, h2p, t) = forward(layers, np.zeros(W3.shape[1]), r)
    m1 = (h1p > 0).astype(np.float32)
    m2 = (h2p > 0).astype(np.float32)
    # J = W3^T diag(m2) W2^T diag(m1) W1^T, scaled by tanh'
    A = np.einsum("ij,nj->nij", W1, m1)            # (N, 4, 64)
    B = np.einsum("nij,jk->nik", A, W2) * m2[:, None, :]  # (N, 4, 64)
    C = np.einsum("nik,kd->nid", B, W3)            # (N, 4, d)
    return np.transpose(C, (0, 2, 1)) * (MAX_SHIFT * (1 - t ** 2))[:, :, None]


def mean_sigma(layers, s, r):
    return forward(layers, s, r)[1].mean(axis=1)


def integrated_gradients(layers, s, r, baseline, steps=64):
    alphas = (np.arange(steps) + 0.5) / steps
    total = np.zeros_like(r)
    for a in alphas:
        total += grad_mean_sigma(layers, s, baseline + a * (r - baseline))
    return (r - baseline) * total / steps


def shapley(layers, s, r, baseline):
    """Exact interventional Shapley values of mean sigma (4 players)."""
    n = r.shape[1]
    phi = np.zeros_like(r)
    from math import factorial
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for k in range(len(others) + 1):
            for S in itertools.combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                x_S = np.tile(baseline, (len(r), 1))
                x_S[:, list(S)] = r[:, list(S)]
                x_Si = x_S.copy()
                x_Si[:, i] = r[:, i]
                phi[:, i] += w * (mean_sigma(layers, s, x_Si) - mean_sigma(layers, s, x_S))
    return phi


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = np.load(os.path.join(_SCRIPT_DIR, "fd002_signals.npz"))
    A = data["A_train"].astype(np.float64)
    rul = data["Y_train"].ravel().astype(np.float64)
    A_test = data["A_test"].astype(np.float64)
    rul_test = data["Y_test"].ravel().astype(np.float64)

    healthy = A[rul >= 125].mean(axis=0)     # baseline: plateau (healthy) regime
    failing = A[rul <= 10].mean(axis=0)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(A), 3000, replace=False)

    summary = {"healthy_baseline": healthy.tolist(), "failing_centroid": failing.tolist(),
               "signal_corr": np.corrcoef(A.T).round(3).tolist(), "runs": {}}
    pdp_grid = np.linspace(-3, 3, 61)
    fig_pdp, ax_pdp = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    fig_rul, ax_rul = plt.subplots(1, 3, figsize=(15, 3.8))
    shap_all, ig_all = {}, {}

    for run in range(3):
        layers, s = load_run(run)
        W1 = layers[0][0]
        ds, sig, (h1p, h2p, _) = forward(layers, s, A)
        rs = {}

        # 1. weights
        rs["W1_input_norms"] = dict(zip(FEATURES, np.linalg.norm(W1, axis=1).round(4).tolist()))
        rs["dead_units_layer1"] = int(((h1p > 0).sum(0) == 0).sum())
        rs["dead_units_layer2"] = int(((h2p > 0).sum(0) == 0).sum())
        rs["base_sigma_mean"] = float((1 / (1 + np.exp(-s))).mean())

        # 2. output geometry
        rs["frac_neg_saturated"] = float((ds <= -SAT_FRAC * MAX_SHIFT).mean())
        rs["frac_pos_saturated"] = float((ds >= SAT_FRAC * MAX_SHIFT).mean())
        ds_c = ds - ds.mean(0)
        U_, S_, Vt = np.linalg.svd(ds_c[sub], full_matrices=False)
        ev = S_ ** 2 / (S_ ** 2).sum()
        rs["delta_s_pca_explained"] = ev[:5].round(4).tolist()
        rs["delta_s_effective_rank"] = float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))
        pc1 = ds_c @ Vt[0]
        if corr(pc1, rul) < 0:
            pc1 = -pc1
        mode_std = ds.std(0)
        rs["n_modes_dynamic(std>0.1)"] = int((mode_std > 0.1).sum())
        rs["corr_modestd_vs_base_s"] = corr(mode_std, s)

        # 3. link to degradation
        msig = sig.mean(1)
        rs["corr_PC1_RUL_train"] = corr(pc1, rul)
        rs["corr_mean_sigma_RUL_train"] = corr(msig, rul)
        msig_test = mean_sigma(layers, s, A_test)
        rs["corr_mean_sigma_RUL_test"] = corr(msig_test, rul_test)
        rs["mean_sigma_healthy"] = float(msig[rul >= 125].mean())
        rs["mean_sigma_failing(RUL<=10)"] = float(msig[rul <= 10].mean())
        rs["corr_PC1_loadings_vs_inputs"] = {f: corr(pc1, A[:, i]) for i, f in enumerate(FEATURES)}

        # 4. attribution (on failing windows vs healthy baseline, and on a random subset)
        g = grad_mean_sigma(layers, s, A[sub])
        rs["mean_abs_grad"] = dict(zip(FEATURES, np.abs(g).mean(0).round(5).tolist()))
        Jd = jacobian_delta_s(layers, A[sub])
        rs["mean_abs_jacobian_delta_s"] = dict(zip(FEATURES, np.abs(Jd).mean((0, 1)).round(4).tolist()))
        fail_idx = np.where(rul <= 20)[0]
        fail_idx = rng.choice(fail_idx, min(2000, len(fail_idx)), replace=False)
        ig = integrated_gradients(layers, s, A[fail_idx], healthy)
        sh = shapley(layers, s, A[fail_idx], healthy)
        gap = mean_sigma(layers, s, A[fail_idx]) - mean_sigma(layers, s, healthy[None])[0]
        rs["IG_mean(fail_vs_healthy)"] = dict(zip(FEATURES, ig.mean(0).round(5).tolist()))
        rs["Shapley_mean(fail_vs_healthy)"] = dict(zip(FEATURES, sh.mean(0).round(5).tolist()))
        rs["Shapley_completeness_err"] = float(np.abs(sh.sum(1) - gap).max())
        rs["IG_completeness_err"] = float(np.abs(ig.sum(1) - gap).mean())
        rs["mean_sigma_drop(fail-healthy)"] = float(gap.mean())
        shap_all[run], ig_all[run] = sh.mean(0), ig.mean(0)

        # 5. partial dependence
        for i, f in enumerate(FEATURES):
            X = np.tile(A[sub], (1, 1))
            pd_vals = []
            for v in pdp_grid:
                X[:, i] = v
                pd_vals.append(mean_sigma(layers, s, X).mean())
            ax_pdp[i].plot(pdp_grid, pd_vals, label=f"run {run}")
            ax_pdp[i].set_title(f"PDP: {f}")
            ax_pdp[i].set_xlabel("standardized value")
        ax_rul[run].scatter(rul[sub], msig[sub], s=3, alpha=0.4)
        ax_rul[run].set_title(f"run {run}: mean σ vs RUL (r={rs['corr_mean_sigma_RUL_train']:.2f})")
        ax_rul[run].set_xlabel("RUL")
        ax_rul[run].set_ylabel("mean σ")

        summary["runs"][run] = rs

    # 6. seed consistency of attribution ranking
    ranks = {r: np.argsort(-np.abs(v)).tolist() for r, v in shap_all.items()}
    summary["shapley_rank_by_run"] = {r: [FEATURES[i] for i in v] for r, v in ranks.items()}

    ax_pdp[0].set_ylabel("mean σ (avg over data)")
    ax_pdp[0].legend()
    fig_pdp.tight_layout()
    fig_pdp.savefig(os.path.join(OUT_DIR, "pdp_mean_sigma.png"), dpi=150)
    fig_rul.tight_layout()
    fig_rul.savefig(os.path.join(OUT_DIR, "mean_sigma_vs_rul.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(4)
    for k, run in enumerate(shap_all):
        ax.bar(x + (k - 1) * 0.27, shap_all[run], width=0.27, label=f"run {run}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, FEATURES)
    ax.set_ylabel("Shapley Δ mean σ")
    ax.set_title("Who drives σ shrinkage? (failing RUL≤20 vs healthy baseline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "shapley_mean_sigma.png"), dpi=150)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
