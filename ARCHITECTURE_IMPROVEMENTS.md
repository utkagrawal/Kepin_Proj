# Koopman-Enhanced Physics-Informed Network (KePIN): Architecture Improvements

This document outlines the architectural enhancements made to the original KePIN model, specifically focusing on the introduction of the **Conditioned Koopman Operator**.

## 1. The Original Architecture Limitation
The original KePIN architecture utilized a static, one-size-fits-all Koopman operator ($K$) to predict future states in the latent space. While effective for simple systems, it assumed that the underlying physical rules (represented by the eigenvalues of the Koopman matrix) remained constant across all machines and operating environments.

## 2. The Improvement: Parameterized (Conditioned) Koopman Operator
To address this limitation, we introduced a **Conditioning Network** (`condition_net`) into the architecture. 

Rather than using a single fixed transition matrix $K$ for all predictions, the improved architecture takes a **Condition Vector** as input. This vector contains external, real-time operating conditions. The conditioning network processes these conditions to dynamically predict a perturbation ($\Delta s$) to the singular values of the Koopman matrix.

Mathematically, instead of a static spectral decomposition:
$$ K = U \cdot \text{diag}(s) \cdot V^T $$

We now have a dynamically conditioned decomposition:
$$ s(\mu) = s + \Delta s(\mu) $$
$$ K(\mu) = U \cdot \text{diag}(s(\mu)) \cdot V^T $$
*(Where $\mu$ is the condition vector.)*

## 3. Physical Parameters Used
For datasets with complex operating regimes (like C-MAPSS FD002 and FD004), we pass a 3-dimensional condition vector (`condition_dim=3`) into the network. These 3 dimensions correspond to the physical engine settings provided in the dataset:
1. **Altitude**
2. **Mach Number** (Speed)
3. **Throttle Resolver Angle** (TRA)

## 4. Key Benefits
* **Personalized Physics:** The model dynamically shifts its internal physics rules on the fly depending on how fast, high, and hard the engine is being run.
* **Higher Accuracy in Multi-Regime Data:** By adapting to specific operating conditions, the predictions become significantly more robust and accurate for engines experiencing highly variable environments. 

## 5. Full System Architecture

For context, here is the high-level architecture of the KePIN model incorporating the Koopman Operator (and the Conditioned variant):

```
Input (B, T, d)  +  Condition Vector (μ)
  │                   │
  ├─ ResConv1D + Squeeze-and-Excitation  ×N blocks
  │                   │
  ├─ Bidirectional LSTM
  │                   │
  ├─ Multi-Head Attention
  │                   │
  ├─ SVD-Parameterised (Conditioned) Koopman Operator
  │      K(μ) = U · diag(s(μ)) · Vᵀ     (latent_dim × latent_dim)
  │      ├─ 1-step prediction: ẑ_{t+1} = K(μ) ẑ_t
  │      ├─ Multi-step rollout: ẑ_{t+k} = K(μ)^k ẑ_t
  │      └─ Spectral features:  |λ|, ∠λ, Re(λ), Im(λ)
  │
  ├─ Concatenation  [encoder_out ‖ spectral_features]
  │
  └─ Deep Prediction Head (256 → 128 → 64, skip connections)
       └─ ŷ  (RUL or forecast)
```
