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
from extract_regime_features import extract_raw_features

def build_autoencoder(input_dim: int, latent_dim: int = 12):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(16, activation='relu')(inputs)
    latent = tf.keras.layers.Dense(latent_dim, activation='linear')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(latent)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    encoder = tf.keras.Model(inputs=inputs, outputs=latent)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model, encoder

def extract_causal_features(df, unit_col, cycle_col, feature_cols, seq_len=30, ref_cycles=10, last_only=False):
    X_feats = []
    
    for unit in sorted(df[unit_col].unique()):
        unit_df = df[df[unit_col] == unit].sort_values(by=cycle_col)
        feats = unit_df[feature_cols].values
        
        # Calculate reference mean (first N cycles)
        ref_mean = np.mean(feats[:min(ref_cycles, len(feats))], axis=0)
        
        if last_only:
            if len(feats) >= seq_len:
                window = feats[-seq_len:]
                raw_feats = extract_raw_features(window, ref_mean)
                X_feats.append(raw_feats)
        else:
            for i in range(len(feats) - seq_len + 1):
                window = feats[i:i + seq_len]
                raw_feats = extract_raw_features(window, ref_mean)
                X_feats.append(raw_feats)
                
    return np.array(X_feats, dtype=np.float32)

def prepare_conditioned_data(dataset_key="CMAPSS_FD003"):
    config_path = os.path.join(_CODE_DIR, "datasets_kepin_config.json")
    with open(config_path) as f:
        all_configs = json.load(f)
    
    idx_map = {"CMAPSS_FD001": 0, "CMAPSS_FD002": 1, "CMAPSS_FD003": 2, "CMAPSS_FD004": 3}
    ds_config = all_configs[idx_map[dataset_key]]
    
    for key in ("train_path", "test_path", "test_rul_path"):
        if ds_config.get(key):
            ds_config[key] = os.path.join(_CODE_DIR, ds_config[key])
            
    # Load via GDS to get the exact X_train_4d, Y_train, X_test_4d, Y_test
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    
    # 3D
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)
    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _ = apply_ema_smoothing(X_test, alpha=ema_alpha)
    
    print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")
    
    # Now load raw data to compute causal features (just like GDS and extract_regime_features do)
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
    
    # Extract causal features
    print("Extracting causal features for train...")
    causal_train = extract_causal_features(train_df, unit_col, cycle_col, feature_cols, seq_len=30, last_only=False)
    print("Extracting causal features for test...")
    causal_test = extract_causal_features(test_df, unit_col, cycle_col, feature_cols, seq_len=30, last_only=True)
    
    print(f"causal_train shape: {causal_train.shape}")
    print(f"causal_test shape: {causal_test.shape}")
    
    assert causal_train.shape[0] == X_train.shape[0], "Mismatch in train windows!"
    assert causal_test.shape[0] == X_test.shape[0], "Mismatch in test windows!"
    
    # Standardize causal features
    c_mean = np.mean(causal_train, axis=0)
    c_std = np.std(causal_train, axis=0) + 1e-8
    causal_train_scaled = (causal_train - c_mean) / c_std
    causal_test_scaled = (causal_test - c_mean) / c_std
    
    # Train Autoencoder to get r
    print("Training autoencoder...")
    tf.random.set_seed(42)
    np.random.seed(42)
    model, encoder = build_autoencoder(input_dim=causal_train.shape[1], latent_dim=12)
    model.fit(causal_train_scaled, causal_train_scaled, epochs=20, batch_size=256, verbose=0)
    
    R_train_raw = encoder.predict(causal_train_scaled, batch_size=512)
    R_test_raw = encoder.predict(causal_test_scaled, batch_size=512)

    from sklearn.preprocessing import StandardScaler
    scaler_r = StandardScaler()
    R_train = scaler_r.fit_transform(R_train_raw).astype(np.float32)
    R_test = scaler_r.transform(R_test_raw).astype(np.float32)
    
    print(f"R_train shape: {R_train.shape}")
    print(f"R_test shape: {R_test.shape}")
    
    # Save everything
    out_file = "fd003_conditioned_data.npz"
    np.savez(out_file, 
             X_train=X_train, Y_train=Y_train, R_train=R_train,
             X_test=X_test, Y_test=Y_test, R_test=R_test)
    
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    prepare_conditioned_data()
