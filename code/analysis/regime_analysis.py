import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
import seaborn as sns

def load_fd002():
    path = "C-MAPSS-Data/train_FD002.txt"
    col_names = ["id", "cycle", "setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]
    df = pd.read_csv(path, sep=r'\s+', header=None, names=col_names)
    
    # Calculate piecewise-linear RUL capped at 125
    rul = pd.DataFrame(df.groupby("id")["cycle"].max()).reset_index()
    rul.columns = ["id", "max_cycle"]
    df = df.merge(rul, on=["id"], how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=125)
    df.drop("max_cycle", axis=1, inplace=True)
    return df

def main():
    out_dir = "experiments_result/ablation_conditioning/phase1"
    os.makedirs(out_dir, exist_ok=True)
    
    df = load_fd002()
    
    settings = ["setting1", "setting2", "setting3"]
    sensors = [f"s{i}" for i in range(1, 22)]
    
    X_set = df[settings].values
    
    # 1. K-Means for k=4..8
    print("Running K-means for K=4..8 on 3 settings...")
    k_res = []
    for k in range(4, 9):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_set)
        
        # Subsample for silhouette to speed up
        idx = np.random.choice(len(X_set), 10000, replace=False)
        sil = silhouette_score(X_set[idx], labels[idx])
        k_res.append({"k": k, "inertia": kmeans.inertia_, "silhouette": sil})
        
    k_df = pd.DataFrame(k_res)
    k_df.to_csv(f"{out_dir}/kmeans_k_evaluation.csv", index=False)
    print(k_df)
    
    # Run K=6 for ground truth regime labels
    kmeans_6 = KMeans(n_clusters=6, random_state=42, n_init=10)
    regime_labels = kmeans_6.fit_predict(X_set)
    df["regime"] = regime_labels
    
    # Plot 3D scatter or pairplot of settings
    print("Plotting settings clustering scatter...")
    plt.figure(figsize=(10, 8))
    sns.pairplot(df[settings + ["regime"]], hue="regime", palette="tab10", plot_kws={'alpha':0.5, 's':10})
    plt.savefig(f"{out_dir}/settings_clustering_scatter.png", dpi=300)
    plt.close('all')
    
    # 2. 2-Subset ARI
    print("Evaluating 2-subsets...")
    subsets = [("setting1", "setting2"), ("setting1", "setting3"), ("setting2", "setting3")]
    ari_res = []
    for sub in subsets:
        km_sub = KMeans(n_clusters=6, random_state=42, n_init=10)
        lbl_sub = km_sub.fit_predict(df[list(sub)].values)
        ari = adjusted_rand_score(regime_labels, lbl_sub)
        ari_res.append({"Subset": f"{sub[0]}_{sub[1]}", "ARI": ari})
        
    ari_df = pd.DataFrame(ari_res)
    ari_df.to_csv(f"{out_dir}/subset_ari.csv", index=False)
    print(ari_df)
    
    # 3 & 4. Mutual Information and ANOVA
    print("Computing MI and ANOVA for settings and sensors...")
    
    # Subsample for MI calculation speed
    idx = np.random.choice(len(df), 10000, replace=False)
    X_sample = df.iloc[idx]
    
    features = settings + sensors
    
    mi_regime = mutual_info_classif(X_sample[features], X_sample["regime"], random_state=42)
    mi_rul = mutual_info_regression(X_sample[features], X_sample["RUL"], random_state=42)
    
    f_stat_regime, p_regime = f_classif(df[features], df["regime"])
    f_stat_rul, p_rul = f_regression(df[features], df["RUL"])
    
    feat_df = pd.DataFrame({
        "Feature": features,
        "MI_Regime": mi_regime,
        "MI_RUL": mi_rul,
        "F_Regime": f_stat_regime,
        "F_RUL": f_stat_rul
    })
    
    feat_df.to_csv(f"{out_dir}/feature_metrics.csv", index=False)
    
    # Analyze Sensors
    sensor_df = feat_df[feat_df["Feature"].isin(sensors)].copy()
    
    # Rank sensors
    # High regime discriminative (MI_Regime), low RUL informative (MI_RUL)
    # Metric: MI_Regime - MI_RUL (or ratio, but difference is safer for 0 denom)
    sensor_df["Regime_Bias"] = sensor_df["MI_Regime"] - sensor_df["MI_RUL"]
    sensor_df = sensor_df.sort_values(by="Regime_Bias", ascending=False)
    
    top_5 = sensor_df.head(5)
    print("Top 5 regime-discriminative (low-degradation) sensors:")
    print(top_5)
    
    # Bar chart for Sensors
    plt.figure(figsize=(12, 6))
    x = np.arange(len(sensor_df))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, sensor_df["MI_Regime"], width, label='MI with Regime')
    rects2 = ax.bar(x + width/2, sensor_df["MI_RUL"], width, label='MI with RUL')
    
    ax.set_ylabel('Mutual Information')
    ax.set_title('Sensor Informativeness: Regime vs RUL')
    ax.set_xticks(x)
    ax.set_xticklabels(sensor_df["Feature"], rotation=45)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig(f"{out_dir}/sensor_MI_ranking.png", dpi=300)
    plt.close('all')
    
    # 6. Write Summary
    summary_text = f"""# Phase 1 Summary: Regime Analysis on FD002

## Settings Subsets
The Adjusted Rand Index (ARI) for the 2-setting subsets compared to the full 3-setting 6-cluster baseline is as follows:
{ari_df.to_markdown(index=False)}

*(Conclusion on redundancy: Depending on the ARI values above, if one subset gives near 1.0 ARI, the missing setting is highly redundant for regime identification. Commonly in C-MAPSS, Setting 3 (TRA) is highly correlated with others or only has discrete values that perfectly map to the others.)*

## Top Regime-Discriminative Sensors
We computed Mutual Information (MI) with the Regime labels and the Remaining Useful Life (RUL) target. Sensors that are highly regime-discriminative but provide little direct degradation information are prime candidates for providing latent conditioning information to the model without directly leaking health status.

The top 5 sensors ranked by their "Regime Bias" (MI_Regime - MI_RUL) are:
{top_5[["Feature", "MI_Regime", "MI_RUL"]].to_markdown(index=False)}
"""
    
    with open(f"{out_dir}/summary.md", "w") as f:
        f.write(summary_text)
        
    print("Phase 1 complete. Summary written.")

if __name__ == "__main__":
    main()
