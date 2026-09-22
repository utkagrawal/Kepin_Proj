import os
import json
import numpy as np
import pandas as pd
from typing import Tuple

def extract_raw_features(window_data: np.ndarray, ref_mean: np.ndarray) -> np.ndarray:
    """
    Computes drift, velocity, and instability for a single window.
    
    Args:
        window_data: (seq_len, F)
        ref_mean: (F,) reference mean of first N cycles
        
    Returns:
        features: (3 * F,) vector
    """
    seq_len, n_features = window_data.shape
    
    # 1. Drift: distance between this window's mean and the reference mean
    window_mean = np.mean(window_data, axis=0)
    drift = window_mean - ref_mean
    
    # 2. Instability: rolling variance over the recent window
    instability = np.var(window_data, axis=0)
    
    # 3. Velocity: slope of linear trend over the window
    # fit a line y = mx + c.  m = cov(x,y)/var(x)
    x = np.arange(seq_len)
    x_mean = np.mean(x)
    x_var = np.var(x) * seq_len # Denominator for slope
    
    # Center x and y
    x_centered = x - x_mean
    y_centered = window_data - window_mean
    
    # Calculate slope (m)
    # Sum of (x_i - x_mean) * (y_i - y_mean)
    cov = np.sum(x_centered[:, None] * y_centered, axis=0)
    velocity = cov / (x_var + 1e-8)
    
    # Combine into a single vector (3 * F,)
    return np.concatenate([drift, velocity, instability])

def build_regime_dataset(data_dir: str, config_path: str, dataset_key: str = "CMAPSS_FD001", seq_len: int = 30, ref_cycles: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads CMAPSS data and extracts raw causal features.
    
    Returns:
        X_feats: (total_windows, 3*F)
        Y_rul: (total_windows, 1)
        Y_elapsed: (total_windows, 1) - normalized elapsed cycles
        unit_ids: (total_windows, 1)
    """
    import sys
    # Add Kepin_code/code to path to use its utilities
    sys.path.append(os.path.join(data_dir, 'Kepin_code', 'code'))
    
    with open(os.path.join(data_dir, 'Kepin_code', 'code', config_path)) as f:
        all_configs = json.load(f)
    
    # Find config
    idx_map = {"CMAPSS_FD001": 0, "CMAPSS_FD002": 1, "CMAPSS_FD003": 2, "CMAPSS_FD004": 3}
    ds_config = all_configs[idx_map[dataset_key]]
    
    train_path = os.path.join(data_dir, 'Kepin_code', 'code', ds_config['train_path'])
    
    # Load Train Data
    column_names = ds_config.get("column_names")
    df = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python")
    
    if column_names and df.shape[1] > len(column_names):
        df = df.iloc[:, :len(column_names)]
    if column_names:
        df.columns = column_names
        
    unit_col = ds_config.get("unit_col", "unit_id")
    cycle_col = ds_config.get("cycle_col", "cycle")
    target_col = ds_config.get("target_col", "RUL")
    feature_cols = ds_config.get("feature_cols")
    
    if feature_cols is None:
        exclude = {unit_col, cycle_col, target_col}
        feature_cols = [c for c in df.columns if c not in exclude]
        
    # Standardize features globally before extracting trends (keeps units comparable)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Add RUL if missing
    if target_col not in df.columns or target_col == "RUL":
        target_col = "RUL"
        max_cycles = df.groupby(unit_col)[cycle_col].transform("max")
        df[target_col] = max_cycles - df[cycle_col]
        
    # Cap RUL at 125 for evaluation (standard practice)
    rul_cap = 125.0
    df[target_col] = df[target_col].clip(upper=rul_cap)
    
    X_feats = []
    Y_rul = []
    Y_elapsed = []
    u_ids = []
    
    # Process per unit
    for unit in sorted(df[unit_col].unique()):
        unit_df = df[df[unit_col] == unit].sort_values(by=cycle_col)
        feats = unit_df[feature_cols].values
        ruls = unit_df[target_col].values
        cycles = unit_df[cycle_col].values
        max_c = np.max(cycles)
        
        # Calculate reference mean (first N cycles)
        ref_mean = np.mean(feats[:min(ref_cycles, len(feats))], axis=0)
        
        # Sliding window extraction
        for i in range(len(feats) - seq_len + 1):
            window = feats[i:i + seq_len]
            rul = ruls[i + seq_len - 1]
            elapsed = cycles[i + seq_len - 1] / max_c  # Normalized elapsed cycles
            
            raw_feats = extract_raw_features(window, ref_mean)
            
            X_feats.append(raw_feats)
            Y_rul.append(rul)
            Y_elapsed.append(elapsed)
            u_ids.append(unit)
            
    X_feats = np.array(X_feats, dtype=np.float32)
    Y_rul = np.array(Y_rul, dtype=np.float32).reshape(-1, 1)
    Y_elapsed = np.array(Y_elapsed, dtype=np.float32).reshape(-1, 1)
    u_ids = np.array(u_ids, dtype=np.int32).reshape(-1, 1)
    
    print(f"Extracted {X_feats.shape[0]} windows. Shape of causal features: {X_feats.shape}")
    return X_feats, Y_rul, Y_elapsed, u_ids

if __name__ == "__main__":
    X, RUL, ELAPSED, UID = build_regime_dataset(".", "datasets_kepin_config.json", dataset_key="CMAPSS_FD003")
    np.savez("fd003_regime_features.npz", X=X, RUL=RUL, ELAPSED=ELAPSED, UID=UID)
    print("Saved to fd003_regime_features.npz")
