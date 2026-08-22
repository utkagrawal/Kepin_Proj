# Conditioning Ablation Study: Final Report

## Objective
The objective of this ablation study was to systematically identify the most effective conditioning strategy for the Koopman Operator on the multi-regime C-MAPSS FD002 dataset. We sought to answer whether the raw continuous 3-setting vector (Altitude, Mach Number, Throttle Resolver Angle) was optimal, or if latent embedding, setting subsets, or regime-discriminative sensors provided a better conditional representation.

## Phase 1 Findings: Regime & Sensor Evidence
In Phase 1, we analyzed the raw data and established the following:
1.  **Setting Redundancy**: 
    - The Adjusted Rand Index (ARI) for `setting1` + `setting2` vs the full 6-cluster baseline was `1.0`.
    - The ARI for `setting1` + `setting3` was `1.0`.
    - The ARI for `setting2` + `setting3` was `0.73`. 
    - This indicated that `setting3` (TRA) and `setting2` (Mach) carry sufficient, if not redundant, information to classify the operational regimes, while `setting1` (Altitude) might be noisier or redundant.
2.  **Top Regime-Discriminative Sensors**: We identified sensors `s1, s18, s5, s6, s21` as highly regime-discriminative but carrying very little RUL degradation signal (high MI with Regime, near-zero MI with RUL).

## Phase 3 & 4: Screening Results (50 Epochs, Single Seed)
We screened multiple conditioning strategies against each other using a reduced 50-epoch budget to identify the clear winner. 

| Rank | Variant | RMSE |
|:---:|:---|:---|
| **1** | `raw2_bc` (Settings 2, 3) | **15.4219** |
| **2** | `hybrid` (`raw2_bc` + `regime_embed`) | 15.4736 |
| **3** | `raw2_ac` (Settings 1, 3) | 15.4966 |
| **4** | `regime_embed` (Latent KMeans) | 15.6756 |
| **5** | `raw2_ab` (Settings 1, 2) | 15.7593 |
| **6** | `sensor_topk_k5` (Top 5 Sensors)| 15.9748 |

**Winning Strategy**: `raw2_bc` (using only Mach Number and Throttle Resolver Angle) proved to be the most effective conditioning input. Dropping `setting1` (Altitude) actually *improves* the model's ability to condition the Koopman eigenvalues, likely by removing redundant noise and forcing the network to rely on the cleanest regime indicators.

## Phase 5: Formal Confirmation (3-Run Ensemble, 250 Epochs)

### Final Evaluation Table
| Model / Strategy | FD002 Test RMSE | NASA Score | Notes |
|:---|:---:|:---:|:---|
| Literature (BiLSTM) | 15.61 | - | Best standard RNN baseline |
| KePIN (Baseline `raw3`) | **14.5673** | - | Original conditioned model using all 3 settings |
| KePIN (`raw2_bc`) | 14.7986 | N/A | Phase 5 confirmation ensemble |

*(Run `python plot_ablation_results.py` in this directory to generate a comparative bar chart).*

## Conclusion & Future Work
While `raw2_bc` achieved an impressive 15.4219 during the rapid 50-epoch screening phase (outperforming all other variants), the full 250-epoch 3-run ensemble converged to **14.7986**. This falls slightly short of the original `raw3` baseline (**14.5673**). 

This indicates that while `setting1` (Altitude) may seem redundant or noisy during early training, its subtle variance still provides marginal long-term capacity for the model when trained fully to convergence. 

**Next Steps**: Since the ablation did not conclusively beat the full-settings baseline, `raw3` remains the state-of-the-art for KePIN. Evaluating the transferability of this conditioning approach to the even more complex **FD004** dataset remains an open follow-up.
