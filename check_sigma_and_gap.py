import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS

def evaluate_sigma_and_gap():
    # 1. Load Data
    data_path = os.path.join(_SCRIPT_DIR, "fd003_signals.npz")
    loaded_data = np.load(data_path)
    X_train = loaded_data["X_train"].astype(np.float32)
    Y_train = loaded_data["Y_train"].astype(np.float32)
    A_train = loaded_data["A_train"].astype(np.float32)
    
    A_test = loaded_data["A_test"].astype(np.float32)
    
    # 2. Configs
    seq_len = X_train.shape[1]
    n_feat = X_train.shape[2]
    n_train = X_train.shape[0]
    
    CFG = CMAPSS_CONFIGS["CMAPSS_FD003"]
    arch_config = CFG["arch_override"].copy()
    arch_config["kernels"] = [min(k, seq_len) for k in arch_config["kernels"]]
    arch_config["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch_config["kernels"]]

    # 3. Build Model
    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, A_train.shape[1]))
    model((dummy_x, dummy_r), training=False)

    # 4. Load Weights
    results_dirs = sorted([d for d in os.listdir(_SCRIPT_DIR) if d.startswith("results_fd003_signal_")], key=lambda x: os.path.getmtime(os.path.join(_SCRIPT_DIR, x)), reverse=True)
    dir_a = [d for d in results_dirs if "_a_" in d][0]
    weights_path = os.path.join(_SCRIPT_DIR, dir_a, "kepin_CMAPSS_FD003_run0.weights.h5")
    model.load_weights(weights_path)
    
    # 5. Compute Sigma Saturation
    # We will use test data for the distribution
    s = model.koopman.s.numpy() # shape (latent_dim,)
    delta_s = 2.0 * tf.nn.tanh(model.koopman.condition_net_raw(tf.constant(A_test))).numpy()
    sigma = tf.nn.sigmoid(s + delta_s).numpy()
    
    # Flatten and analyze
    sigma_flat = sigma.flatten()
    sat_low = np.sum(sigma_flat < 0.05) / len(sigma_flat) * 100
    sat_high = np.sum(sigma_flat > 0.95) / len(sigma_flat) * 100
    sat_total = sat_low + sat_high
    
    # Percentiles
    p5, p25, p50, p75, p95 = np.percentile(sigma_flat, [5, 25, 50, 75, 95])
    
    # 6. Compute Train RMSE
    # Batch predict to avoid OOM
    batch_size = 256
    preds = []
    for i in range(0, len(X_train), batch_size):
        x_b = X_train[i:i+batch_size]
        r_b = A_train[i:i+batch_size]
        pred = model((x_b, r_b), training=False)[0] # Extract the RUL prediction from tuple
        preds.append(pred.numpy())
    preds = np.concatenate(preds, axis=0).flatten()
    
    train_rmse = np.sqrt(mean_squared_error(Y_train.flatten(), preds))
    test_rmse = 11.0088787 # from the json
    gap = test_rmse - train_rmse
    
    print("=========================================================")
    print(" SIGNAL A: TRAIN VS TEST RMSE GAP")
    print("=========================================================")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test RMSE:  {test_rmse:.4f}")
    print(f"Gap:        {gap:.4f} (Test - Train)")
    print("\n=========================================================")
    print(" SIGNAL A: SIGMA SATURATION DISTRIBUTION (TEST SET)")
    print("=========================================================")
    print(f"Total Sigma values: {len(sigma_flat)}")
    print(f"Mean Sigma: {np.mean(sigma_flat):.4f}")
    print(f"Percentiles:")
    print(f"  5th:  {p5:.4f}")
    print(f" 25th:  {p25:.4f}")
    print(f" 50th:  {p50:.4f}")
    print(f" 75th:  {p75:.4f}")
    print(f" 95th:  {p95:.4f}")
    print(f"\nSaturation (Close to boundaries):")
    print(f"  < 0.05: {sat_low:.2f}%")
    print(f"  > 0.95: {sat_high:.2f}%")
    print(f"  Total Saturated: {sat_total:.2f}%")
    print("=========================================================")

if __name__ == "__main__":
    evaluate_sigma_and_gap()
