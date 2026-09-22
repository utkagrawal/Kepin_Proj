# FD003 Saturation-Fixed Regime-Conditioned Koopman: Final Results

This document presents the final evaluation of the regime-conditioned Koopman architecture on CMAPSS FD003 after applying two critical stability fixes:
1. **Wider, Normalized Bottleneck**: The autoencoder bottleneck was widened to 12 dimensions, and the resulting regime vector $r$ was normalized (zero mean, unit variance) using training set statistics.
2. **Bounded Output**: The raw output of `condition_net` was strictly bounded using $\Delta s = 2.0 \cdot \tanh(\text{raw})$, ensuring that the final eigenvalue shifts could not blindly saturate the sigmoid function into numerical oblivion, while perfectly preserving the 0-initialized warm-start property.

## 1. RMSE Comparison

The implementation successfully trained without crashing, but the resulting performance is actively worse than the static baseline.

| Model | Test RMSE |
| :--- | :--- |
| **Static Baseline** (from prior runs) | **11.42** |
| Original Unfixed Conditioned (Saturated) | 11.46 |
| **Fixed Conditioned (12D, Bounded)** | **12.72** |

## 2. Saturation Check (The Fix Worked)

To determine whether the bounded $\tanh$ actually solved the saturation problem (rather than simply causing the model to break in a different way), we measured the fraction of $\sigma(r)$ values across the test set that fall within a "healthy," non-saturated dynamic range of `[0.05, 0.95]`. 

| Model | Healthy $\sigma(r)$ Fraction |
| :--- | :--- |
| Original Unfixed Conditioned | 24.44% |
| **Fixed Conditioned (Bounded)** | **97.92%** |

The original model was heavily saturated (over 75% of eigenvalues were pushed to the absolute extremes). The bounded fix completely cured this, successfully forcing the network to operate dynamically within a healthy, responsive range.

## 3. $|\Delta s|$ Magnitude Check

- **Mean $|\Delta s|$**: 1.6027
- **Mean Std($\Delta s$)**: 0.4794

This confirms the output is mathematically capped by the $2.0$ bound and behaves safely, varying dynamically across the test set without spiraling to infinity.

## 4. Dimension Utilization (12D Bottleneck)

With the wider 12-dimensional bottleneck, we checked whether the network simply collapsed to using 2 or 3 dimensions or whether it utilized the newly provided bandwidth.

| Dimension | Variance | Correlation with True RUL |
| :--- | :--- | :--- |
| Dim 1 | 1.0924 | -0.5949 |
| Dim 2 | 0.8535 | -0.6866 |
| Dim 3 | 1.6568 | -0.4799 |
| Dim 4 | 0.9153 | -0.3473 |
| Dim 5 | 0.5507 | -0.1315 |
| Dim 6 | 1.2921 | 0.2803 |
| Dim 7 | 0.9174 | 0.0755 |
| Dim 8 | 1.1846 | 0.7308 |
| Dim 9 | 0.9739 | 0.5247 |
| Dim 10 | 1.6474 | -0.4085 |
| Dim 11 | 0.6895 | 0.7932 |
| Dim 12 | 0.7149 | 0.2593 |

The network heavily utilized almost every dimension. Nearly all dimensions exhibit variance near $1.0$, and the majority demonstrate strong, distinct correlations (both positive and negative) with the degradation trajectory. The wide bottleneck is functioning exactly as intended.

## 5. Stability Diagnostics

The Cayley transform construction for dynamic orthogonal matrices remains numerically flawless:
- $||U^T U - I||_F$: $3.9692 \times 10^{-6}$
- $||V^T V - I||_F$: $3.9388 \times 10^{-6}$

---

## 6. FINAL VERDICT

> [!IMPORTANT]
> **Conditioning does not universally benefit all degradation datasets.**

The mechanical problems with the regime-conditioned architecture have been unequivocally solved. The model utilizes a rich, wide input vector, generates highly dynamic and healthy shifts in the Koopman eigenvalues, and remains strictly orthogonal and numerically stable.

Despite these perfect mechanical diagnostics, the model scores a **12.72** RMSE—significantly worse than the static baseline's **11.42**. 

Because we have a healthy, non-saturated $\sigma$ distribution and a model that operates exactly as mathematically intended, this is the honest, final answer for FD003. We accept this outcome: regime-conditioning simply does not improve performance on this specific dataset, likely because FD003's degradation physics do not vary enough between engines to warrant the massive capacity increase of a dynamic operator. This result will be reported honestly alongside the FD004 results (which *did* show improvement) to conclude that dynamic regime conditioning is highly dataset-dependent. No further architectural tweaks will be pursued.
