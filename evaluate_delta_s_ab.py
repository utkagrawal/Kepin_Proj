import os
import sys
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model

from kepin_cmapss_optimized import CMAPSS_CONFIGS

def evaluate_signal(signal_name, weights_path, signal_data, X_train):
    # 1. Model Configs
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]
    
    CFG = CMAPSS_CONFIGS["CMAPSS_FD003"]
    arch_config = CFG["arch_override"].copy()
    arch_config["kernels"] = [min(k, seq_len) for k in arch_config["kernels"]]
    arch_config["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch_config["kernels"]]

    # 2. Build model and load weights
    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, signal_data.shape[1]))
    model((dummy_x, dummy_r), training=False)

    model.load_weights(weights_path)
    
    # 4. Extract s and delta_s
    s = model.koopman.s.numpy() # shape: (LatentDim,) -> (96,)
    delta_s = 2.0 * tf.nn.tanh(model.koopman.condition_net_raw(tf.constant(signal_data))).numpy() # shape: (N_test, 96)
    
    # 5. Compute Statistics
    mean_abs_s = np.mean(np.abs(s))
    mean_abs_delta_s = np.mean(np.abs(delta_s))
    
    std_delta_s = np.std(delta_s, axis=0) # shape (96,)
    mean_std_delta_s = np.mean(std_delta_s)
    
    relative_scale = (mean_abs_delta_s / mean_abs_s) * 100 if mean_abs_s != 0 else 0
    
    print(f"=========================================================")
    print(f" EVALUATION FOR SIGNAL {signal_name}")
    print(f"=========================================================")
    print(f"Base Vector |s| (mean abs magnitude):     {mean_abs_s:.6f}")
    print(f"Conditioning |delta_s| (mean abs mag):    {mean_abs_delta_s:.6f}")
    print(f"Conditioning Scale relative to |s|:       {relative_scale:.2f}%")
    print(f"Std Dev of delta_s across windows (mean): {mean_std_delta_s:.6f}")
    print(f"Max Std Dev across any single dimension:  {np.max(std_delta_s):.6f}")
    print(f"=========================================================\n")


def main():
    # Load signals
    data_path = os.path.join(_SCRIPT_DIR, "fd003_signals.npz")
    loaded_data = np.load(data_path)
    X_train = loaded_data["X_train"].astype(np.float32)
    A_test = loaded_data["A_test"].astype(np.float32)
    B_test = loaded_data["B_test"].astype(np.float32)
    
    # Find weights paths
    # Assuming latest folders
    results_dirs = sorted([d for d in os.listdir(_SCRIPT_DIR) if d.startswith("results_fd003_signal_")], key=lambda x: os.path.getmtime(os.path.join(_SCRIPT_DIR, x)), reverse=True)
    dir_a = [d for d in results_dirs if "_a_" in d][0]
    dir_b = [d for d in results_dirs if "_b_" in d][0]
    
    weights_a = os.path.join(_SCRIPT_DIR, dir_a, "kepin_CMAPSS_FD003_run0.weights.h5")
    weights_b = os.path.join(_SCRIPT_DIR, dir_b, "kepin_CMAPSS_FD003_run0.weights.h5")
    
    evaluate_signal("A (4D Moments)", weights_a, A_test, X_train)
    evaluate_signal("B (1D Residual)", weights_b, B_test, X_train)

if __name__ == "__main__":
    main()
