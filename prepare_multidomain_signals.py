import os
import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import GenericTimeSeriesDataset as GDS
from extract_statistical_moments import compute_window_moments

def convert_4d_to_3d(X):
    if X.ndim == 4 and X.shape[2] == 1:
        return X[:, :, 0, :]
    return X

def prepare_signals(dataset_key):
    print(f"\n--- Preparing Signals for {dataset_key} ---")
    config_path = os.path.join(_CODE_DIR, "datasets_all_config.json")
    with open(config_path) as f:
        all_configs = json.load(f)
    
    # Find config
    ds_config = None
    for config in all_configs:
        if config.get("name") == dataset_key:
            ds_config = config
            break
    
    if ds_config is None:
        raise ValueError(f"Dataset {dataset_key} not found in {config_path}")
    
    # Adjust paths if needed
    for key in ("train_path", "test_path", "test_rul_path", "csv_path"):
        if ds_config.get(key) and not os.path.isabs(ds_config[key]):
            ds_config[key] = os.path.join(_CODE_DIR, ds_config[key])
            
    # Load via GDS to get the exact X_train, Y_train
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    
    # 3D
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)
    
    print("Extracting statistical moments for train...")
    A_train = np.array([compute_window_moments(X_train[i]) for i in range(X_train.shape[0])], dtype=np.float32)
    
    print("Extracting statistical moments for test...")
    A_test = np.array([compute_window_moments(X_test[i]) for i in range(X_test.shape[0])], dtype=np.float32)
    
    print(f"A_train shape: {A_train.shape}")
    print(f"A_test shape: {A_test.shape}")
    
    assert A_train.shape[0] == X_train.shape[0], "Mismatch in train windows!"
    assert A_test.shape[0] == X_test.shape[0], "Mismatch in test windows!"
    
    # Standardize Signal A
    scaler_A = StandardScaler()
    A_train_scaled = scaler_A.fit_transform(A_train).astype(np.float32)
    A_test_scaled = scaler_A.transform(A_test).astype(np.float32)
    
    # Save everything
    out_file = f"{dataset_key.lower()}_signals.npz"
    np.savez(out_file, 
             X_train=X_train, Y_train=Y_train, 
             X_test=X_test, Y_test=Y_test,
             A_train=A_train_scaled, A_test=A_test_scaled)
    
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    for ds in ["Jena_Climate", "Cylinder_Wake", "Building_Energy"]:
        prepare_signals(ds)
