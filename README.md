# Regime-Conditioned KePIN (RC-KePIN)

This branch contains the implementation of the **Dynamic Koopman Operator** via Regime-Conditioning for the KePIN (Koopman-enhanced Physics-Informed Neural Network) architecture.

## Overview
Traditional Koopman operators in time-series forecasting assume a static, time-invariant dynamic matrix. While this works well for stable systems, it fails to accurately model systems undergoing severe physical degradation, where the "rules of physics" (the transition dynamics) change over time.

This project introduces a **Dynamic Koopman Operator**, where the underlying Koopman eigenvalues ($\sigma$) are dynamically shifted ($\Delta s$) based on the real-time statistical regime of the system.

## Methodology

### 1. Regime Extraction (Signal A)
Instead of feeding the raw multivariate time-series into the condition network, we extract macro-level statistical moments over a sliding time window (e.g., $w=30$).
For each window, we compute:
- **Mean**
- **Variance**
- **Skewness**
- **Kurtosis**

These 4 aggregated statistical moments form **Signal A**, which acts as a proxy for the physical "regime" of the engine.

### 2. Dynamic Eigenvalue Shift ($\Delta s$)
Signal A is passed into a dedicated condition neural network within the Koopman layer. This network outputs a shift vector, $\Delta s$, bounded by a `tanh` activation function. 
This shift is applied to the base eigenvalues $s$ before the sigmoid activation:
$$ \sigma = \text{sigmoid}(s + \Delta s) $$

### 3. Physical Intuition
As an engine degrades, its variance and skewness increase. The condition network detects this regime change and outputs a shift to $\Delta s$. This dynamically shrinks the singular values ($\sigma$), which mathematically **accelerates the decay rate** of the latent system state, perfectly mirroring the accelerated physical degradation of a failing engine.

## Usage

*   `extract_statistical_moments.py`: Core logic for computing Signal A.
*   `Kepin_code/code/koopman_module.py`: The updated Koopman layer featuring the condition network.
*   `plot_delta_s_history.py`: Script to visualize the dynamic shifting of the eigenvalues over an engine's lifespan.
