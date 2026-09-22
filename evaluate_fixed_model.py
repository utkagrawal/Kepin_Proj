import os
import sys
import json
import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
import glob

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model

def get_latest_results_dir():
    dirs = glob.glob(os.path.join(_SCRIPT_DIR, "results_fd003_cond_*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    return sorted(dirs, key=os.path.getmtime)[-1] if dirs else None

def main():
    # 1. Model Configs
    seq_len = 30
    n_feat = 16
    n_train = 21820
    
    arch_config = {
        "tier": "medium",
        "n_blocks": 4,
        "filters": [64, 128, 128, 256],
        "kernels": [11, 7, 5, 3],
        "latent_dim": 96,
        "lstm_units": 96,
        "n_heads": 4,
        "head_key_dim": 24,
        "dropout": 0.3,
        "rollout": 3,
        "spectral_k": 5
    }

    # ==========================================
    # EVALUATE FIXED MODEL (12D)
    # ==========================================
    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, 12))
    model((dummy_x, dummy_r), training=False)

    results_dir = get_latest_results_dir()
    print(f"Loading weights from {results_dir}...")
    weights_path = os.path.join(results_dir, "kepin_CMAPSS_FD003_run0.weights.h5")
    model.load_weights(weights_path)

    # Load 12D R_test
    data_path = os.path.join(_SCRIPT_DIR, "fd003_conditioned_data.npz")
    loaded_data = np.load(data_path)
    R_test = loaded_data["R_test"].astype(np.float32)
    Y_test = loaded_data["Y_test"].astype(np.float32)
    
    s = model.koopman.s.numpy() # shape: (96,)
    
    cond_out_raw = model.koopman.condition_net(tf.constant(R_test))
    delta_s = 2.0 * tf.math.tanh(cond_out_raw).numpy()
    
    sigma = tf.nn.sigmoid(s + delta_s).numpy()
    
    # 2. SATURATION CHECK
    healthy = np.sum((sigma >= 0.05) & (sigma <= 0.95))
    total = sigma.size
    fraction_healthy = healthy / total
    
    # 3. DELTA_S STATS
    mean_abs_s = np.mean(np.abs(s))
    mean_abs_delta_s = np.mean(np.abs(delta_s))
    std_delta_s = np.std(delta_s, axis=0)
    mean_std_delta_s = np.mean(std_delta_s)
    
    # 3b. DIMENSION UTILIZATION CHECK
    print("=========================================================")
    print(" 3b. DIMENSION UTILIZATION CHECK (r variance and corr)")
    print("=========================================================")
    for i in range(12):
        var_r = np.var(R_test[:, i])
        corr_r_y, _ = pearsonr(R_test[:, i], Y_test.flatten())
        print(f"  Dimension {i+1:2d}: Var = {var_r:.4f}, Corr w/ RUL = {corr_r_y:.4f}")
    
    # 4. STABILITY
    M_U = model.koopman.M_U
    M_V = model.koopman.M_V
    I_mat = tf.eye(M_U.shape[0])
    A_U = M_U - tf.transpose(M_U)
    U = tf.linalg.solve(I_mat + A_U, I_mat - A_U)
    A_V = M_V - tf.transpose(M_V)
    V = tf.linalg.solve(I_mat + A_V, I_mat - A_V)
    
    U_err = tf.norm(tf.matmul(U, U, transpose_a=True) - I_mat)
    V_err = tf.norm(tf.matmul(V, V, transpose_a=True) - I_mat)
    
    print("\n=========================================================")
    print(" FINAL VERIFICATION RESULTS")
    print("=========================================================")
    print(f"Sigma(r) Healthy [0.05, 0.95] Fraction: {fraction_healthy:.2%}")
    print(f"Mean |delta_s|:                         {mean_abs_delta_s:.4f}")
    print(f"Mean Std(delta_s) across test set:      {mean_std_delta_s:.4f}")
    print(f"||U^T U - I||_F:                        {U_err.numpy():.4e}")
    print(f"||V^T V - I||_F:                        {V_err.numpy():.4e}")

if __name__ == "__main__":
    main()
