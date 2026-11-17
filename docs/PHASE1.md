# Phase 1 checkpoint (15 November 2026)

**Thesis question:** how can the Koopman operator in KePIN be made adaptive to the engine's health, and does that help
prediction or explanation? Full easy-language guide: `docs/KePIN_Project_Explained.pdf`; report: `docs/REPORT.md`;
technical log: `dynamic_koopman/EXPERIMENT_LOG.md`.

## What phase 1 established

| # | Step | Result |
|---|---|---|
| 1 | Original KePIN, examined | Its Koopman part is unused (removing it: no significant change; spectral features identical for every engine); the original code picks its model on the test set. |
| 2 | Condition-aware input layer | Beats KePIN by 0.9–1.8 RMSE on FD002/FD004 (paper protocol). An input fix, not a Koopman result. |
| 3 | Adaptive Koopman from the 30-flight window (gate, mixture, per-engine rule, wear model) | No gain; knockout 0.00. The encoder already straightens the wear curve (5× → 1.8×). |
| 4 | Hypernetwork Koopman (Hyper-LR-PINN style: shared basis P, per-engine decay ρ and rotation θ) | Ignored with window context; used once it sees engine age. |
| 5 | Health from the engine's whole history → hypernetwork | All 4 C-MAPSS datasets: −2 to −5 RMSE vs KePIN (fair test). Control C (same health straight to the predictor) ties B everywhere. |
| 6 | Explainability (arm A, health only through Koopman) | Health falls in 100% of engines (no RUL labels); the rule keeps more modes as health falls (corr −0.96..−0.99); breaking the Koopman path costs +8 to +16 RMSE; per-engine "health dial". |
| 7 | Cross-domain (KePIN's other datasets) | No gain: Cylinder Wake all tie; KePIN best or tied on Building Energy and Jena; SPY unpredictable for all. The history adds no new information there. |

**Unifying finding:** an adaptive Koopman operator matters only when it carries information the predictor cannot
otherwise see (window context: 0.00, age: +2.4, whole-life health: +8 to +16 RMSE when removed).

## Final C-MAPSS numbers (ensemble RMSE, 3 seeds)

| Dataset | KePIN (fair / paper) | A: health only | B: health + age | C: control |
|---|---|---|---|---|
| FD001 | 14.32 / 12.92 | 13.98 / 12.28 | **11.68** / 10.59 | 11.85 / **10.55** |
| FD002 | 15.40 / 15.00 | 13.70 / 13.09 | **11.77 / 11.03** | 11.99 / 11.04 |
| FD003 | 13.12 / 11.56 | 13.14 / 11.61 | **11.06 / 9.90** | 11.31 / 10.11 |
| FD004 | 17.84 / 17.11 | 15.21 / 14.51 | 12.92 / **11.84** | **12.64** / 11.88 |

Reviewer check: gradient-boosted trees on the 42 history features + age alone give 10.45 / 13.00 / 10.68 / 12.70
(FD001–FD004) — the accuracy gain belongs to the history features, not to KePIN or Koopman.

## Cross-domain (ensemble RMSE, fair test, original units)

| Dataset | KePIN | static K | hyper (window) | A | B | C |
|---|---|---|---|---|---|---|
| Cylinder Wake | 0.0051 | 0.0051 | 0.0051 | 0.0051 | 0.0051 | 0.0051 |
| Building Energy | **0.0921** | 0.0936 | 0.0959 | 0.0938 | 0.0930 | 0.0934 |
| Jena Climate (°C) | 5.24 | 5.34 | 5.25 | 5.26 | **5.22** | 5.65 |
| SPY Stock (days, R² < 0 for all) | 43.1 | 38.4 | 33.4 | 34.4 | 37.2 | 33.5 |

## Open items for phase 2

1. Add the cross-domain results and the boosted-tree baseline to the PDF and report.
2. Anchor the direction of the health number (flipped in 9 of 72 C-MAPSS models).
3. Make Koopman necessary: wear model + history-health hypernetwork rule.
4. Optional: 5 seeds, N-CMAPSS, sensor-level interpretation of the failure-regime modes.

Model weights are not in git; the full local snapshot (code, data caches, weights, logs) is
`RCKePIN_checkpoints/phase1_RCKePIN_2026-11-15.tar.gz` on the cluster.
