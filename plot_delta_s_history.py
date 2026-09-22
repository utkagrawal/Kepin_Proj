import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model
from kepin_cmapss_optimized import CMAPSS_CONFIGS

def main():
    # Load signals
    data_path = os.path.join(_SCRIPT_DIR, "fd003_signals.npz")
    loaded_data = np.load(data_path)
    X_train = loaded_data["X_train"].astype(np.float32)
    # USE A_TRAIN BECAUSE TEST SET ONLY HAS LAST WINDOW (NO HISTORY)
    A_train = loaded_data["A_train"].astype(np.float32)
    Y_train = loaded_data["Y_train"].astype(np.float32)
    
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
    dummy_r = tf.zeros((1, A_train.shape[1]))
    model((dummy_x, dummy_r), training=False)

    results_dirs = sorted([d for d in os.listdir(_SCRIPT_DIR) if d.startswith("results_fd003_signal_")], key=lambda x: os.path.getmtime(os.path.join(_SCRIPT_DIR, x)), reverse=True)
    dir_a = [d for d in results_dirs if "_a_" in d][0]
    weights_path = os.path.join(_SCRIPT_DIR, dir_a, "kepin_CMAPSS_FD003_run0.weights.h5")
    
    try:
        model.load_weights(weights_path)
    except Exception:
        pass # Ignore complex64 warning if weights loaded
    
    # 4. Extract delta_s for the TRAIN set
    delta_s = 2.0 * tf.nn.tanh(model.koopman.condition_net_raw(tf.constant(A_train))).numpy()
    
    # Find the boundaries of the first engine in the training set
    # RUL is decreasing. A jump means a new engine.
    diffs = np.diff(Y_train.flatten())
    first_engine_end = np.where(diffs > 0)[0][0] + 1
    
    engine_delta_s = delta_s[:first_engine_end]
    engine_rul = Y_train[:first_engine_end].flatten()
    
    mean_delta_s = np.mean(engine_delta_s, axis=1)
    
    # Calculate singular values (sigma)
    s = model.koopman.s.numpy()
    engine_sigma = tf.nn.sigmoid(s + engine_delta_s).numpy()
    mean_sigma = np.mean(engine_sigma, axis=1)
    
    cycles = np.arange(len(mean_delta_s))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(cycles, mean_delta_s, label='Mean Shift (Δs)', color='blue', linewidth=2)
    ax1.plot(cycles, mean_sigma, label='Mean Singular Value (σ)', color='purple', linewidth=2)
    
    ax2 = ax1.twinx()
    ax2.plot(cycles, engine_rul, label='Remaining Useful Life (RUL)', color='green', alpha=0.3)
    ax2.set_ylabel('RUL (Cycles)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title("Dynamic Koopman: Eigenvalue Shift (Δs) and Singular Values (σ)")
    ax1.set_xlabel("Elapsed Cycles (Time)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(True, alpha=0.3)
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='center left')
    
    plot_path = os.path.join(_SCRIPT_DIR, "delta_s_history.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {plot_path}")

if __name__ == "__main__":
    main()
