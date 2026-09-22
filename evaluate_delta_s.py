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

    # 2. Build model and load weights
    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, 3))
    model((dummy_x, dummy_r), training=False)

    weights_path = os.path.join(_SCRIPT_DIR, "results_fd003_cond_2791", "kepin_CMAPSS_FD003_run0.weights.h5")
    model.load_weights(weights_path)

    # 3. Load R_test
    data_path = os.path.join(_SCRIPT_DIR, "fd003_conditioned_data.npz")
    loaded_data = np.load(data_path)
    R_test = loaded_data["R_test"].astype(np.float32)
    
    # 4. Extract s and delta_s
    s = model.koopman.s.numpy() # shape: (LatentDim,) -> (96,)
    delta_s = model.koopman.condition_net(tf.constant(R_test)).numpy() # shape: (N_test, 96)
    
    # 5. Compute Statistics
    
    # Mean absolute magnitude of s
    mean_abs_s = np.mean(np.abs(s))
    
    # Mean absolute magnitude of delta_s (overall)
    mean_abs_delta_s = np.mean(np.abs(delta_s))
    
    # Standard deviation of delta_s across windows
    std_delta_s = np.std(delta_s, axis=0) # shape (96,)
    mean_std_delta_s = np.mean(std_delta_s)
    
    # Relative scale
    relative_scale = (mean_abs_delta_s / mean_abs_s) * 100 if mean_abs_s != 0 else 0
    
    # Correlation between norm of r and norm of delta_s
    r_norms = np.linalg.norm(R_test, axis=1)
    delta_s_norms = np.linalg.norm(delta_s, axis=1)
    correlation, p_val = pearsonr(r_norms, delta_s_norms)
    
    # Correlation per component (just averaging the R2 for a simple check)
    corr_per_dim = []
    for i in range(delta_s.shape[1]):
        # correlate with r_norms for simplicity, or with the original r components
        c, _ = pearsonr(r_norms, delta_s[:, i])
        if not np.isnan(c):
            corr_per_dim.append(np.abs(c))
    mean_abs_corr_per_dim = np.mean(corr_per_dim) if corr_per_dim else 0.0

    print("=========================================================")
    print(" CONDITION NET EVALUATION (DELTA S)")
    print("=========================================================")
    print(f"Base Vector |s| (mean abs magnitude):     {mean_abs_s:.6f}")
    print(f"Conditioning |delta_s| (mean abs mag):    {mean_abs_delta_s:.6f}")
    print(f"Conditioning Scale relative to |s|:       {relative_scale:.2f}%")
    print(f"Std Dev of delta_s across windows (mean): {mean_std_delta_s:.6f}")
    print(f"Max Std Dev across any single dimension:  {np.max(std_delta_s):.6f}")
    print(f"Correlation (|r| vs |delta_s|):           {correlation:.4f} (p={p_val:.4e})")
    print(f"Mean Abs Correlation (dim-wise vs |r|):   {mean_abs_corr_per_dim:.4f}")
    print("=========================================================")

if __name__ == "__main__":
    main()
