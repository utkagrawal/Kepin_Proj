import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
import tensorflow as tf
from keras import layers, models

def build_autoencoder(input_dim: int, latent_dim: int = 3):
    """Builds a simple MLP autoencoder."""
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation='relu')(encoder_input)
    latent_output = layers.Dense(latent_dim, activation='linear', name='latent')(x)
    
    decoder_input = layers.Dense(16, activation='relu')(latent_output)
    decoder_output = layers.Dense(input_dim, activation='linear')(decoder_input)
    
    autoencoder = models.Model(encoder_input, decoder_output)
    encoder = models.Model(encoder_input, latent_output)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def evaluate_and_report(data_path="fd001_regime_features.npz", docs_dir="docs"):
    os.makedirs(docs_dir, exist_ok=True)
    
    # 1. Load Data
    data = np.load(data_path)
    X = data['X']  # (N, 42)
    RUL = data['RUL'].flatten()  # (N,)
    ELAPSED = data['ELAPSED'].flatten()  # (N,)
    UID = data['UID'].flatten()  # (N,)
    
    # 2. Train Autoencoder
    input_dim = X.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim=3)
    
    print("Training regime encoder (autoencoder)...")
    # Quick training since it's just a simple MLP
    autoencoder.fit(X, X, epochs=20, batch_size=128, validation_split=0.2, verbose=1)
    
    # 3. Extract Regimes
    r_latent = encoder.predict(X)  # (N, 3)
    
    # 4. Extract dominant component via PCA
    pca = PCA(n_components=1)
    r_1d = pca.fit_transform(r_latent).flatten()
    
    # Ensure monotonic trend aligns negatively with RUL (just a sign flip if needed for readability)
    if pearsonr(r_1d, RUL)[0] > 0:
        r_1d = -r_1d
        
    # 5. Calculate Metrics
    # Correlations
    r_rul_pearson, _ = pearsonr(r_1d, RUL)
    r_rul_spearman, _ = spearmanr(r_1d, RUL)
    
    r_elapsed_pearson, _ = pearsonr(r_1d, ELAPSED)
    r_elapsed_spearman, _ = spearmanr(r_1d, ELAPSED)
    
    # Smoothness (variance of first derivative per unit)
    step_variances = []
    for u in np.unique(UID):
        mask = UID == u
        r_u = r_1d[mask]
        if len(r_u) > 1:
            diffs = np.diff(r_u)
            step_variances.append(np.var(diffs))
    smoothness_score = np.mean(step_variances)
    
    # 6. Generate Plot
    plt.figure(figsize=(10, 6))
    sample_units = np.random.choice(np.unique(UID), size=5, replace=False)
    for u in sample_units:
        mask = UID == u
        r_u = r_1d[mask]
        t_u = ELAPSED[mask]
        plt.plot(t_u, r_u, marker='o', markersize=3, label=f'Unit {u}')
        
    plt.title('Extracted Regime Signal (Dominant Component) over Time')
    plt.xlabel('Normalized Elapsed Cycles')
    plt.ylabel('Regime Signal (PCA of $r$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(docs_dir, 'regime_trajectories.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # 7. Write Report
    report_path = os.path.join(docs_dir, 'PHASE2_REGIME_EXTRACTION_QUALITY.md')
    
    # Verdict logic
    abs_rul_corr = abs(r_rul_spearman)
    abs_elapsed_corr = abs(r_elapsed_spearman)
    if abs_rul_corr > 0.5 and abs_elapsed_corr > 0.5:
        verdict = "**MEANINGFUL**. The extracted signal shows a strong correlation with degradation and elapsed time, indicating a plausible monotonic trend from healthy to near-failure."
    else:
        verdict = "**WEAK/NOISY**. The extracted signal does not strongly correlate with degradation or time, suggesting it may be extracting noise or requires a supervised monotonicity loss."
        
    report_content = f"""# Phase 2: Regime Extraction Quality

## Evaluation Metrics

### Correlation Analysis
*   **Correlation with True RUL**: 
    *   Pearson: `{r_rul_pearson:.4f}`
    *   Spearman: `{r_rul_spearman:.4f}`
*   **Correlation with Normalized Elapsed Cycles**:
    *   Pearson: `{r_elapsed_pearson:.4f}`
    *   Spearman: `{r_elapsed_spearman:.4f}`

*(Note: High correlation with elapsed cycles suggests the regime vector is strongly acting as a time-index proxy, which provides a meaningful monotonic trend but may not capture complex failure modes beyond "how far into the run we are".)*

### Smoothness
*   **Step-to-step variance (mean per unit)**: `{smoothness_score:.6f}`
*(Lower is smoother. Erratic jumps would result in a high variance.)*

## Trajectory Visualization
The plot below shows the dominant component of the extracted 3D regime vector $r$ for a random sample of 5 engines over their lifetime.

![Regime Trajectories](./regime_trajectories.png)

## Verdict
{verdict}
"""

    with open(report_path, 'w') as f:
        f.write(report_content)
        
    print(f"Evaluation complete. Report saved to {report_path}")

if __name__ == "__main__":
    evaluate_and_report()
