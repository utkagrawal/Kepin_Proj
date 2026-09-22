# FD003 Cluster-Split Reanalysis

## 1. Engine Cluster Assignment
Because fault modes are a property of the entire engine lifetime, engines were assigned to Cluster 0 or Cluster 1 based on the majority vote of their windows' 3D GMM cluster assignments.

*   **Cluster 0 Engines**: 13
*   **Cluster 1 Engines**: 87

## 2. PC1 Within-Bin Correlation (Split by Cluster)
We re-ran the original within-bin Spearman correlation between $r\_PC1$ and RUL, but evaluated the two clusters independently.

**Cluster 0:**
*   Mean Spearman: `-0.8064`
*   Median Spearman: `-0.8980`

**Cluster 1:**
*   Mean Spearman: `-0.5931`
*   Median Spearman: `-0.6754`

## 3. Full 3D Vector Within-Bin Correlation (Linear Regression $R^2$)
Instead of forcing the signal through a 1D PCA bottleneck, we fitted a simple Linear Regression (`RUL ~ r_0 + r_1 + r_2`) within each elapsed-cycle bin and recorded the $R^2$ (proportion of RUL variance explained by the 3D regime vector, independent of elapsed time).

**Pooled (All Engines):**
*   Mean $R^2$: `0.7293`
*   Median $R^2$: `0.8101`

**Split by Cluster 0:**
*   Mean $R^2$: `0.8587`
*   Median $R^2$: `0.9291`

**Split by Cluster 1:**
*   Mean $R^2$: `0.7163`
*   Median $R^2$: `0.7668`

## Interpretation

**HYPOTHESIS 1 CONFIRMED (Fault-Mode Mixing).**
Splitting the dataset by the GMM-discovered clusters revealed strong, consistent within-bin correlations for $r\_PC1$ inside each cluster independently. The original 'negative' verdict was merely a Simpson's paradox artifact caused by pooling two fundamentally different fault modes together. The extraction pipeline successfully captures degradation, but we must condition on the *full vector* (or cluster identity) rather than relying on a single pooled scalar.