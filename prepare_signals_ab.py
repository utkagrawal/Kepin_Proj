import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import convert_4d_to_3d
from kepin_training import apply_ema_smoothing
import GenericTimeSeriesDataset as GDS
from extract_statistical_moments import compute_window_moments
from extract_local_linear_residuals import compute_local_linear_residual

def extract_signals_ab(df, unit_col, cycle_col, feature_cols, seq_len=30, last_only=False):
    """Extracts Signal A (4D) and Signal B (1D) for each sliding window."""
    signal_a = []
    signal_b = []
    
    for unit in sorted(df[unit_col].unique()):
        unit_df = df[df[unit_col] == unit].sort_values(by=cycle_col)
        feats = unit_df[feature_cols].values
        
        if last_only:
            if len(feats) >= seq_len:
                window = feats[-seq_len:]
                signal_a.append(compute_window_moments(window))
                signal_b.append(compute_local_linear_residual(window))
        else:
            for i in range(len(feats) - seq_len + 1):
                window = feats[i:i + seq_len]
                signal_a.append(compute_window_moments(window))
                signal_b.append(compute_local_linear_residual(window))
                
    return np.array(signal_a, dtype=np.float32), np.array(signal_b, dtype=np.float32)

def prepare_signals(dataset_key="CMAPSS_FD003"):
    config_path = os.path.join(_CODE_DIR, "datasets_kepin_config.json")
    with open(config_path) as f:
        all_configs = json.load(f)
    
    idx_map = {"CMAPSS_FD001": 0, "CMAPSS_FD002": 1, "CMAPSS_FD003": 2, "CMAPSS_FD004": 3}
    ds_config = all_configs[idx_map[dataset_key]]
    
    for key in ("train_path", "test_path", "test_rul_path"):
        if ds_config.get(key):
            ds_config[key] = os.path.join(_CODE_DIR, ds_config[key])
            
    # Load via GDS to get the exact X_train, Y_train
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    
    # 3D
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    
    # Load raw dataframe to compute causal features without ema-smoothing destroying variances
    train_path = ds_config['train_path']
    test_path = ds_config['test_path']
    
    def load_df(path):
        column_names = ds_config.get("column_names")
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        if column_names and df.shape[1] > len(column_names):
            df = df.iloc[:, :len(column_names)]
        if column_names:
            df.columns = column_names
        return df

    train_df = load_df(train_path)
    test_df = load_df(test_path)
    
    unit_col = ds_config.get("unit_col", "unit_id")
    cycle_col = ds_config.get("cycle_col", "cycle")
    target_col = ds_config.get("target_col", "RUL")
    feature_cols = ds_config.get("feature_cols")
    
    if feature_cols is None:
        exclude = {unit_col, cycle_col, target_col}
        feature_cols = [c for c in train_df.columns if c not in exclude]
        
    # Standardize globally (fit on train)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    # Extract Signals A and B
    print("Extracting signals A and B for train...")
    A_train, B_train = extract_signals_ab(train_df, unit_col, cycle_col, feature_cols, seq_len=30, last_only=False)
    
    print("Extracting signals A and B for test...")
    A_test, B_test = extract_signals_ab(test_df, unit_col, cycle_col, feature_cols, seq_len=30, last_only=True)
    
    print(f"A_train shape: {A_train.shape}, B_train shape: {B_train.shape}")
    print(f"A_test shape: {A_test.shape}, B_test shape: {B_test.shape}")
    
    assert A_train.shape[0] == X_train.shape[0], "Mismatch in train windows!"
    assert A_test.shape[0] == X_test.shape[0], "Mismatch in test windows!"
    
    # Standardize Signal A
    scaler_A = StandardScaler()
    A_train_scaled = scaler_A.fit_transform(A_train).astype(np.float32)
    A_test_scaled = scaler_A.transform(A_test).astype(np.float32)
    
    # Standardize Signal B
    scaler_B = StandardScaler()
    B_train_scaled = scaler_B.fit_transform(B_train).astype(np.float32)
    B_test_scaled = scaler_B.transform(B_test).astype(np.float32)
    
    # Save everything
    out_file = "fd003_signals.npz"
    np.savez(out_file, 
             X_train=X_train, Y_train=Y_train, 
             X_test=X_test, Y_test=Y_test,
             A_train=A_train_scaled, A_test=A_test_scaled,
             B_train=B_train_scaled, B_test=B_test_scaled)
    
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    prepare_signals()
