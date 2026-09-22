import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

def build_autoencoder(input_dim: int, latent_dim: int = 3):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    encoder = tf.keras.Model(inputs=inputs, outputs=latent)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model, encoder

def check_bimodality():
    print("Loading data...")
    data = np.load("fd003_regime_features.npz")
    X = data['X']
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    print("Training encoder to get r...")
    # Train quickly just to get r
    tf.random.set_seed(42)
    np.random.seed(42)
    model, encoder = build_autoencoder(input_dim=X.shape[1], latent_dim=3)
    model.fit(X_scaled, X_scaled, epochs=20, batch_size=256, verbose=0)
    
    r = encoder.predict(X_scaled, batch_size=512)
    
    # 1. 1D PCA (r_PC1)
    pca = PCA(n_components=1)
    r_pc1 = pca.fit_transform(r).flatten()
    
    # Fit GMMs for r_PC1
    r_pc1_2d = r_pc1.reshape(-1, 1)
    gmm1_1d = GaussianMixture(n_components=1, random_state=42).fit(r_pc1_2d)
    gmm2_1d = GaussianMixture(n_components=2, random_state=42).fit(r_pc1_2d)
    
    bic1_1d = gmm1_1d.bic(r_pc1_2d)
    bic2_1d = gmm2_1d.bic(r_pc1_2d)
    aic1_1d = gmm1_1d.aic(r_pc1_2d)
    aic2_1d = gmm2_1d.aic(r_pc1_2d)
    
    # 2. 3D r vector
    gmm1_3d = GaussianMixture(n_components=1, random_state=42).fit(r)
    gmm2_3d = GaussianMixture(n_components=2, random_state=42).fit(r)
    
    bic1_3d = gmm1_3d.bic(r)
    bic2_3d = gmm2_3d.bic(r)
    aic1_3d = gmm1_3d.aic(r)
    aic2_3d = gmm2_3d.aic(r)
    
    # Separation metrics for 3D GMM
    labels_3d = gmm2_3d.predict(r)
    probs_3d = gmm2_3d.predict_proba(r)
    avg_confidence = np.mean(np.max(probs_3d, axis=1))
    
    means = gmm2_3d.means_
    dist = np.linalg.norm(means[0] - means[1])
    # db_score = davies_bouldin_score(r, labels_3d) # Lower is better separation
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(r_pc1, bins=50, kde=True, stat="density", color="blue", alpha=0.6)
    plt.title('Distribution of Dominant Regime Signal (r_PC1) on FD003')
    plt.xlabel('r_PC1')
    plt.ylabel('Density')
    
    # Overlay GMM 2-component fit
    x_range = np.linspace(r_pc1.min(), r_pc1.max(), 1000).reshape(-1, 1)
    logprob = gmm2_1d.score_samples(x_range)
    pdf = np.exp(logprob)
    plt.plot(x_range, pdf, color='red', linestyle='--', linewidth=2, label='2-Component GMM Fit')
    plt.legend()
    
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/fd003_bimodality_plot.png")
    
    report = f"""# FD003 Bimodality Check

## 1. 1D Dominant Signal (r_PC1)
We evaluated whether the primary component of the regime vector shows a bimodal distribution (indicative of discovering multiple fault modes unconditionally).

*   **1-Component GMM**: BIC = `{bic1_1d:.1f}`, AIC = `{aic1_1d:.1f}`
*   **2-Component GMM**: BIC = `{bic2_1d:.1f}`, AIC = `{aic2_1d:.1f}`

*(Lower BIC/AIC indicates a better model fit. A significantly lower score for the 2-component model strongly supports bimodality).*

## 2. Full 3D Regime Vector (r)
We repeated the GMM fitting on the uncompressed 3D latent vector.

*   **1-Component GMM**: BIC = `{bic1_3d:.1f}`, AIC = `{aic1_3d:.1f}`
*   **2-Component GMM**: BIC = `{bic2_3d:.1f}`, AIC = `{aic2_3d:.1f}`

**Cluster Separation (2-Component 3D GMM):**
*   **Distance between Cluster Means**: `{dist:.4f}`
*   **Average Cluster Assignment Confidence (Posterior Prob)**: `{avg_confidence:.4f}` 
*(A confidence near 0.5 means overlapping/unclear clusters; near 1.0 means highly separated/distinct clusters).*

![Bimodality Plot](./fd003_bimodality_plot.png)

## Final Verdict
"""
    if bic2_1d < bic1_1d and avg_confidence > 0.8:
        report += "**CLEAR BIMODALITY DETECTED.** The 2-component Gaussian Mixture Model provides a significantly better fit than a single Gaussian. The 3D clusters are well-separated with high assignment confidence. This strongly suggests the unsupervised encoder successfully discovered the two distinct fault modes present in FD003."
    else:
        report += "**NO CLEAR BIMODALITY.** The distributions appear unimodal or highly overlapping as a single continuum. The unsupervised encoder did not naturally separate the data into two distinct fault-mode clusters."
        
    with open("docs/FD003_BIMODALITY_CHECK.md", "w") as f:
        f.write(report)
        
    print("Done! Report written to docs/FD003_BIMODALITY_CHECK.md")

if __name__ == "__main__":
    check_bimodality()
