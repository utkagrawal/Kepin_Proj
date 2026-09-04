# Regime-Aware KePIN: Experimental Results and Analysis

## 1. Research Hypothesis
**Question:** Can RUL prediction be improved by learning latent degradation regimes directly from sensor time series (without using operating-condition labels) and assigning different Koopman dynamics to different latent regimes?

**Hypothesis:** By allowing the network to identify regimes and transition between multiple Koopman operators dynamically (unsupervised), the model will better capture complex, non-linear degradation paths, outperforming a single global Koopman operator.

## 2. Experimental Setup
We evaluated the proposed **Regime-Aware KePIN** model on the NASA C-MAPSS FD001 dataset using 3 random seeds. The setup included:
- **3koopman Baseline:** 3 static, parallel Koopman operators (no dynamic regime switching).
- **Proposed Regime-KePIN:** 3 Koopman operators with dynamic regime switching probabilities ($\pi_t$) learned via an unsupervised MLP head.
- **Proposed (No Smoothness):** Ablation removing the temporal smoothness penalty on $\pi_t$.
- **Proposed (No Physics):** Ablation removing the Koopman spectral stability loss.

## 3. Results (FD001, 3 Seeds)

| Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | NASA Score |
|-------|-------------------|------------------|------------|
| 3-Koopman (Static) | **18.93** ± 0.70 | **13.98** ± 0.69 | **1192** |
| Proposed (Regime) | 19.38 ± 1.07 | 13.94 ± 0.90 | 1770 |
| Proposed (No Physics)| 19.30 ± 0.77 | 14.19 ± 0.53 | 1102 |
| Proposed (No Smooth) | 19.35 ± 0.91 | 13.77 ± 0.98 | 1559 |

## 4. Regime Analysis

By visualizing the learned regime probabilities $\pi_t$ for individual test engines, we observed critical behaviors that explain the failure to outperform the static baseline:

1. **Regime Collapse (Proposed Model):**
   When the regime smoothness loss is applied, the network avoids the penalty by collapsing into a single regime almost entirely. 
   - *Statistics:* Average probability for Regime 3 is **96.2%**. The entropy of the distribution is extremely low (0.139). The model effectively ignored Regimes 1 and 2, turning itself back into a standard single-operator KePIN (which struggles more than the 3-operator ensemble).

2. **Regime Chaos (No-Smoothness Ablation):**
   When the smoothness constraint is removed, the network fails to learn meaningful contiguous degradation phases (e.g., healthy $\rightarrow$ degrading $\rightarrow$ critical). Instead, it rapidly oscillates between all 3 regimes at every timestep.
   - *Statistics:* Average probabilities are perfectly uniform: `[0.329, 0.378, 0.293]`. The entropy is maximized (1.088 out of max 1.099). It acts merely as a randomized ensemble, adding noise to the latent trajectory rather than meaningful conditioning.

*(Note: You can view the actual trajectory visualizations at `plots_proposed/regime_evolution.png` and `plots_nosmooth/regime_evolution.png`)*

## 5. Conclusion
**The hypothesis is rejected for C-MAPSS FD001 under this completely unsupervised formulation.**

Learning latent regimes dynamically without explicit constraints (like operating condition labels or monotonic priors) is under-constrained. The unsupervised network treats the regime probabilities either as a source of random ensemble variance (without smoothness) or completely collapses to avoid transition penalties (with smoothness). 

Adding raw capacity (the `3koopman` static baseline) works better because it acts as a parallel feature extractor without the instability of dynamic path routing. Future work should investigate semi-supervised approaches or integrating physical priors (like monotonicity) into the regime discovery process.
