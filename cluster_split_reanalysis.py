import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression

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

def run_analysis():
    print("Loading data...")
    data = np.load("fd003_regime_features.npz")
    X = data['X']
    RUL = data['RUL'].flatten()
    UID = data['UID'].flatten()
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_scaled = (X - X_mean) / X_std
    
    # Calculate actual elapsed cycles
    actual_cycles = np.zeros(len(UID))
    for u in np.unique(UID):
        mask = (UID == u)
        num_windows = np.sum(mask)
        actual_cycles[mask] = np.arange(30, 30 + num_windows)
        
    print("Training encoder to get r...")
    tf.random.set_seed(42)
    np.random.seed(42)
    model, encoder = build_autoencoder(input_dim=X.shape[1], latent_dim=3)
    model.fit(X_scaled, X_scaled, epochs=20, batch_size=256, verbose=0)
    
    r = encoder.predict(X_scaled, batch_size=512)
    
    pca = PCA(n_components=1)
    r_pc1 = pca.fit_transform(r).flatten()
    
    print("Assigning clusters per engine...")
    gmm = GaussianMixture(n_components=2, random_state=42).fit(r)
    window_clusters = gmm.predict(r)
    
    # Assign cluster per engine via majority vote
    unit_clusters = {}
    for u in np.unique(UID):
        mask = (UID == u)
        u_clusters = window_clusters[mask]
        majority_cluster = int(np.bincount(u_clusters).argmax())
        unit_clusters[u] = majority_cluster
        
    # Map back to windows
    assigned_clusters = np.array([unit_clusters[u] for u in UID])
    
    df = pd.DataFrame({
        'r_pc1': r_pc1,
        'r_0': r[:, 0],
        'r_1': r[:, 1],
        'r_2': r[:, 2],
        'rul': RUL,
        'cycles': actual_cycles,
        'cluster': assigned_clusters
    })
    
    bins = np.arange(0, df['cycles'].max() + 10, 10)
    df['cycle_bin'] = pd.cut(df['cycles'], bins=bins, right=False)
    
    def get_1d_corrs(df_subset):
        corrs = []
        names = []
        for name, group in df_subset.groupby('cycle_bin', observed=True):
            if len(group) >= 30:
                c, _ = spearmanr(group['r_pc1'], group['rul'])
                if not np.isnan(c):
                    corrs.append(c)
                    names.append(str(name))
        return names, corrs, np.mean(corrs) if corrs else 0, np.median(corrs) if corrs else 0
        
    def get_3d_r2(df_subset):
        r2s = []
        names = []
        for name, group in df_subset.groupby('cycle_bin', observed=True):
            if len(group) >= 30:
                X_3d = group[['r_0', 'r_1', 'r_2']].values
                y = group['rul'].values
                reg = LinearRegression().fit(X_3d, y)
                score = reg.score(X_3d, y) # R^2
                if score >= 0:
                    r2s.append(score)
                    names.append(str(name))
        return names, r2s, np.mean(r2s) if r2s else 0, np.median(r2s) if r2s else 0

    # 1. PC1 per cluster
    c0_names, c0_corrs, c0_mean, c0_med = get_1d_corrs(df[df['cluster'] == 0])
    c1_names, c1_corrs, c1_mean, c1_med = get_1d_corrs(df[df['cluster'] == 1])
    
    # 2. 3D Regression Overall
    all_3d_names, all_3d_r2s, all_3d_mean, all_3d_med = get_3d_r2(df)
    
    # 3. 3D Regression per cluster
    c0_3d_names, c0_3d_r2s, c0_3d_mean, c0_3d_med = get_3d_r2(df[df['cluster'] == 0])
    c1_3d_names, c1_3d_r2s, c1_3d_mean, c1_3d_med = get_3d_r2(df[df['cluster'] == 1])

    report = f"""# FD003 Cluster-Split Reanalysis

## 1. Engine Cluster Assignment
Because fault modes are a property of the entire engine lifetime, engines were assigned to Cluster 0 or Cluster 1 based on the majority vote of their windows' 3D GMM cluster assignments.

*   **Cluster 0 Engines**: {np.sum(np.array(list(unit_clusters.values())) == 0)}
*   **Cluster 1 Engines**: {np.sum(np.array(list(unit_clusters.values())) == 1)}

## 2. PC1 Within-Bin Correlation (Split by Cluster)
We re-ran the original within-bin Spearman correlation between $r\_PC1$ and RUL, but evaluated the two clusters independently.

**Cluster 0:**
*   Mean Spearman: `{c0_mean:.4f}`
*   Median Spearman: `{c0_med:.4f}`

**Cluster 1:**
*   Mean Spearman: `{c1_mean:.4f}`
*   Median Spearman: `{c1_med:.4f}`

## 3. Full 3D Vector Within-Bin Correlation (Linear Regression $R^2$)
Instead of forcing the signal through a 1D PCA bottleneck, we fitted a simple Linear Regression (`RUL ~ r_0 + r_1 + r_2`) within each elapsed-cycle bin and recorded the $R^2$ (proportion of RUL variance explained by the 3D regime vector, independent of elapsed time).

**Pooled (All Engines):**
*   Mean $R^2$: `{all_3d_mean:.4f}`
*   Median $R^2$: `{all_3d_med:.4f}`

**Split by Cluster 0:**
*   Mean $R^2$: `{c0_3d_mean:.4f}`
*   Median $R^2$: `{c0_3d_med:.4f}`

**Split by Cluster 1:**
*   Mean $R^2$: `{c1_3d_mean:.4f}`
*   Median $R^2$: `{c1_3d_med:.4f}`

## Interpretation

"""
    
    # Interpretation logic
    improved_1d = (abs(c0_med) > 0.35 and abs(c1_med) > 0.35)
    improved_3d = (all_3d_med > 0.15) # R^2 > 0.15 is roughly R > ~0.38
    
    if improved_1d:
        report += "**HYPOTHESIS 1 CONFIRMED (Fault-Mode Mixing).**\nSplitting the dataset by the GMM-discovered clusters revealed strong, consistent within-bin correlations for $r\_PC1$ inside each cluster independently. The original 'negative' verdict was merely a Simpson's paradox artifact caused by pooling two fundamentally different fault modes together. The extraction pipeline successfully captures degradation, but we must condition on the *full vector* (or cluster identity) rather than relying on a single pooled scalar."
    elif improved_3d:
        report += "**HYPOTHESIS 2 CONFIRMED (1D Bottleneck).**\nWhile splitting by cluster didn't rescue the 1D $r\_PC1$ signal, using the full 3D vector $r$ directly yielded much stronger predictive power ($R^2$) within each time bin. The issue was that PCA discarded critical degradation information present in the orthogonal dimensions. The pipeline works, but we must condition on the full 3D vector."
    else:
        report += "**NEGATIVE RESULT (No Improvement).**\nNeither splitting by the discovered clusters nor utilizing the full 3D latent vector meaningfully improved the within-bin correlations. The unsupervised extraction approach on FD003 fails to yield a degradation-consistent signal even when controlling for fault modes and dimensionality. We should proceed to a supervised extraction method or skip the regime bottleneck entirely."
        
    os.makedirs("docs", exist_ok=True)
    with open("docs/FD003_CLUSTER_SPLIT_REANALYSIS.md", "w") as f:
        f.write(report)
    print("Done! Saved to docs/FD003_CLUSTER_SPLIT_REANALYSIS.md")

if __name__ == "__main__":
    run_analysis()
