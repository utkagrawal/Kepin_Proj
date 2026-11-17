#!/usr/bin/env python3
"""Writes Part 3 (adaptive Koopman with hypernetworks, 22 Oct - 10 Nov 2026) into FULL_REPORT/REPORT.md,
between the markers <!-- PART3 START --> / <!-- PART3 END --> (inserted before the appendix on first run).
Numbers are computed from runs/ and reports/hyper_explain/, so re-running refreshes them."""
import glob
import json
import os

import numpy as np

DK = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.dirname(DK)
REPORT = os.path.join(REPO, "FULL_REPORT", "REPORT.md")
DSS = ["FD001", "FD002", "FD003", "FD004"]
ARMS = [("svd_static", "KePIN"), ("eig_hyper_hh_mono", "A"), ("eig_hyper_hh_age_rank_mono", "B"),
        ("eig_static_hh_hctx_mono", "C")]


def ens(p, ds, arm):
    fs = sorted(glob.glob(os.path.join(DK, "runs", p, ds, arm, "run*_predictions.npz")))
    d = [np.load(f) for f in fs]
    y = d[0]["y_true"].ravel(); pr = np.mean([x["y_pred"].ravel() for x in d], 0)
    e = pr - y
    return y, pr, float(np.sqrt(np.mean(e ** 2))), float(np.sum(np.where(e < 0, np.exp(-e / 13) - 1, np.exp(e / 10) - 1))), len(fs)


def boot(a, b, seed=0):
    (y, pa), (_, pb) = a, b
    ea, eb = (pa - y) ** 2, (pb - y) ** 2; n = len(y); rng = np.random.default_rng(seed)
    dd = [np.sqrt(ea[i].mean()) - np.sqrt(eb[i].mean()) for i in (rng.integers(0, n, n) for _ in range(4000))]
    return f"{np.sqrt(ea.mean()) - np.sqrt(eb.mean()):+.2f} [{np.percentile(dd, 2.5):+.2f}, {np.percentile(dd, 97.5):+.2f}]"


R = {(p, ds, n): ens(p, ds, arm) for p in ("clean", "paper") for ds in DSS for arm, n in ARMS}
fmt = lambda r: f"{r[2]:.2f} / {r[3]:.0f}" + ("" if r[4] == 3 else f" ({r[4]} seed{'s' if r[4] > 1 else ''})")
EX = {ds: json.load(open(os.path.join(DK, "reports", "hyper_explain", f"{ds}_clean.json"))) for ds in DSS}
av = lambda ds, k: np.mean([r[k] for r in EX[ds]["runs"]])
ko = lambda ds, k: np.mean([r["knockout"][k] - r["knockout"]["normal"] for r in EX[ds]["runs"]])

L = []
w = L.append
w("## Part 3: making the Koopman operator adaptive (22 October – 10 November 2026)\n")
w("**The question:** instead of one Koopman rule for the whole fleet, can each engine get its own rule that depends on "
  "how healthy it is? Does that help accuracy, and does it explain anything? (Full easy-language version with all "
  "figures: `KePIN_Project_Explained.pdf`, sections 13–23.)\n")
w("### Step 19: Adaptive rules built from the 30-flight window (condition-aware KePIN and wear model)\n")
w("- **Health index without labels.** A learned health number h collapses to a constant unless told what health is. "
  "One rule fixes it, the *mono loss*: a later window of the same engine must have a lower h. It never uses RUL labels, "
  "yet h correlates 0.87–0.90 with true remaining life, falls in 100% of engines and ignores the flying condition (R² 0.00).")
w("- **Three adaptive rules** (health gate, mixture of regimes, a rule fitted per engine): none beat the static rule "
  "(FD002: 13.87 static vs 13.97–14.17). Knockout: forcing every engine to the healthy or failing rule changes RMSE by "
  "**0.00**. The predictor reads the hidden state directly, so a rule computed from it adds nothing.")
w("- **Wear model (Koopman is the predictor):** Koopman matters (freezing its fading modes costs +1.5 to +3.0) but "
  "adapting it does not (static 13.68 vs 14.04–14.17; healthy rule for everyone: −0.01).")
w("- **Why:** the encoder straightens the degradation curve. The cleaned sensors speed up 5× near failure; a "
  "self-supervised latent keeps 2.8×, the RUL-trained latent only 1.8× (the first ReLU removes most). On encoders that keep "
  "the curve the gate helped, but its h was not health (R² ≈ 0 with RUL): extra capacity, not adaptation.\n")
w("![Curvature through the encoder](figures/16_curvature_layers.png)\n")
w("**Lesson:** an adaptive Koopman rule fed from the window has nothing new to say. Health information must come from "
  "outside the window, and Koopman must sit on a path the predictor uses.\n")
w("### Step 20: Hypernetwork Koopman in original KePIN\n")
w("A **hypernetwork** is a small network that writes the settings of another part of the model. Following Hyper-LR-PINN "
  "(Cho et al., NeurIPS 2023), the Koopman rule is written as **K = P · diag(ρ<sub>k</sub> R(θ<sub>k</sub>)) · Pᵀ**: the "
  "directions P are shared by all engines, and the hypernetwork (Dense 64 ReLU → Dense, zero-initialised) writes each "
  "engine's 64 fade factors ρ<sub>k</sub> and turn angles θ<sub>k</sub> (and optional on/off gates). The eigenvalues then "
  "differ per engine, so KePIN's 12 spectral features, which were identical for all engines, now carry per-engine "
  "information.\n")
w("![Hypernetwork architecture](figures/17_hyper_architecture.png)\n")
w("First runs (FD002, paper's method): window context 14.72 (knockout **0.00**); window + age 14.39; + on/off gates "
  "**14.04** (knockout +2.4, age shuffled +7.7). **Rule found:** Koopman contributes only when its spectrum carries "
  "information the predictor cannot already see.\n")
w("### Step 21: Health from the engine's whole history\n")
w("- **42 engine-history features** per window, from all flights so far (past only, so usable at test time); each sensor "
  "scaled per flying condition, then *drift since new* (last 10 flights − first 10) and *current rate* (slope of the last "
  "30). Linear fit to remaining life on test engines: R² 0.81 (FD001), 0.81 (FD002), 0.82 (FD003), 0.83 (FD004).")
w("- **Health net** (Dense 32 ReLU → Dense 1 sigmoid) turns them into h, trained only with the mono loss.")
w("- **Three arms:** **A** hypernetwork reads only h (so every change of K is caused by health); **B** reads h + age, with "
  "on/off gates; **C (control)** static K, and h + age go straight into the predictor.\n")
w("![Results on all datasets](figures/29_results_all.png)\n")
for p, title in [("clean", "Fair test"), ("paper", "Paper's method")]:
    w(f"**{title}** (ensemble RMSE / NASA score, 3 seeds):\n")
    w("| Dataset | original KePIN | A: health only | B: health + age | C: control |")
    w("|---|---|---|---|---|")
    for ds in DSS:
        w(f"| {ds} | " + " | ".join(fmt(R[(p, ds, n)]) for _, n in ARMS) + " |")
    w("")
w("**Fair test, bootstrap over test engines (difference [95% range]):**\n")
w("| Dataset | A − KePIN | B − KePIN | C − KePIN | **B − C** |")
w("|---|---|---|---|---|")
for ds in DSS:
    g = lambda n: R[("clean", ds, n)][:2]
    w(f"| {ds} | {boot(g('A'), g('KePIN'))} | {boot(g('B'), g('KePIN'))} | {boot(g('C'), g('KePIN'))} | "
      f"**{boot(g('B'), g('C'))}** |")
w("")
w("- **History health helps on every dataset:** B and C beat original KePIN by about 2–5 RMSE; biggest on FD004 (the "
  "hardest). FD003's gain (≈ −2) has a 95% range touching 0 (only 100 test engines).")
w("- **B ties C on all four datasets.** The hypernetwork Koopman passes the health information on; it is not an extra "
  "source of accuracy.")
w("- **A** helps on the multi-condition datasets (FD002, FD004) but not on FD001/FD003.")
w("- **Caveat:** in 9 of 72 models h came out *flipped* (rises towards failure). Accuracy is unaffected; for explanations "
  "the direction should be anchored. All arm A fair-test models have the right direction.\n")
w("### Step 22: How the hypernetwork makes Koopman explainable (arm A, fair test, all datasets)\n")
w("![The idea](figures/22_hyper_concept.png)\n")
w("In arm A the hypernetwork reads **one number, h**. So the whole Koopman rule of an engine is a function of its health, "
  "and the model's reasoning can be checked in four steps (`analyze_hyper_explain.py`, `reports/hyper_explain/`).\n")
w("![Eigenvalues healthy vs near failure](figures/23_eig_circle.png)\n")
w("![h over life](figures/26_h_trajectories.png)\n")
w("![Life curves](figures/25_life_all.png)\n")
w("| Arm A, fair test (avg of 3 seeds) | h falls over life | corr(h, RUL) | corr(avg ρ, h) | avg ρ healthy → near failure | slow modes (ρ > 0.3) | knockout: same K for all | h shuffled |")
w("|---|---|---|---|---|---|---|---|")
for ds in DSS:
    w(f"| {ds} | {100 * av(ds, 'h_falls_over_life_share'):.0f}% of engines | {av(ds, 'h_corr_rul_capped'):.2f} | "
      f"{av(ds, 'corr_mean_rho_h'):.2f} | {av(ds, 'mean_rho_healthy'):.2f} → {av(ds, 'mean_rho_near_failure'):.2f} | "
      f"{av(ds, 'slow_modes_healthy'):.0f} → {av(ds, 'slow_modes_near_failure'):.0f} of 64 | "
      f"+{ko(ds, 'same K for all engines'):.1f} | +{ko(ds, 'h shuffled'):.1f} |")
w("")
w("![Knockout on all datasets](figures/27_knockout_all.png)\n")
w("![Health dial](figures/24_health_dial.png)\n")
w("1. **The health number is health:** trained without RUL labels, it falls over life in 100% of engines on all four "
  "datasets.")
w("2. **The Koopman rule follows health the same way everywhere:** as h falls, modes fade less (ρ about 0.15 → 0.3–0.4) "
  "and many more modes stay slow (about 0–1 → 20–60 of 64), with a clear *failure regime* at low h.")
w("3. **The prediction goes through the Koopman rule:** breaking that path costs 8–16 RMSE (0.00 in original KePIN).")
w("4. **Health dial (what-if):** keeping an engine's 30 flights fixed and feeding a chosen h from 0 to 1 moves its "
  "prediction smoothly by about 35–95 flights; the rule switches to its failure regime below h ≈ 0.25 (FD002).")
w("- **Opposite to the original hypothesis:** the latent space is squeezed hardest while healthy and kept larger near "
  "failure, on every dataset and seed.")
w("- **Honest caveat:** most modes still lose half their size within one step, so the spectrum is a readable *health "
  "code* more than multi-step dynamics.\n")
w("![Per-mode fade factor vs health](figures/28_mode_heatmaps.png)\n")
w("### Part 3 conclusions\n")
w("1. **An adaptive Koopman operator contributes only when it carries information the predictor cannot otherwise see** "
  "(window context: 0.00; age: +2.4; history health: +8 to +16).")
w("2. **Engine-history health improves KePIN on all four datasets** (≈ −2 to −5 RMSE, fair test).")
w("3. **The health-driven hypernetwork Koopman (arm A) is a consistent, readable explanation** of the model, with a "
  "per-engine what-if tool, but **Koopman is not needed for the accuracy** (B ties C everywhere).")
w("4. **Next:** anchor the direction of h; make Koopman necessary (e.g. wear-model forecast with a history-health "
  "hypernetwork rule).\n")
w("---\n")
body = "\n".join(L)

txt = open(REPORT).read()
S, E = "<!-- PART3 START -->", "<!-- PART3 END -->"
block = f"{S}\n{body}\n{E}\n\n"
if S in txt:
    a, b = txt.index(S), txt.index(E) + len(E)
    txt = txt[:a] + block.rstrip("\n") + txt[b:]
else:
    txt = txt.replace("## Appendix: file map", block + "## Appendix: file map", 1)
open(REPORT, "w").write(txt)
print("written", REPORT, "| FD001 clean KePIN seeds:", R[("clean", "FD001", "KePIN")][4])
