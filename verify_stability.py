import os
import sys
import numpy as np
import tensorflow as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model

def main():
    # Model configs
    seq_len = 30
    n_feat = 16
    n_train = 21820
    
    # Needs to match FD003 medium tier arch
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

    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    # Initialize with dummy input
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, 3))
    model((dummy_x, dummy_r), training=False)

    weights_path = os.path.join(_SCRIPT_DIR, "results_fd003_cond_2791", "kepin_CMAPSS_FD003_run0.weights.h5")
    if not os.path.exists(weights_path):
        print("Weights not found yet...")
        return
        
    model.load_weights(weights_path)

    # 1. Check U and V orthogonality
    M_U = model.koopman.M_U
    M_V = model.koopman.M_V
    I_mat = tf.eye(M_U.shape[0])
    A_U = M_U - tf.transpose(M_U)
    U = tf.linalg.solve(I_mat + A_U, I_mat - A_U)
    A_V = M_V - tf.transpose(M_V)
    V = tf.linalg.solve(I_mat + A_V, I_mat - A_V)
    
    I = tf.eye(U.shape[0])
    
    U_err = tf.norm(tf.matmul(U, U, transpose_a=True) - I)
    V_err = tf.norm(tf.matmul(V, V, transpose_a=True) - I)
    
    print(f"||U^T U - I||_F = {U_err.numpy():.4e}")
    print(f"||V^T V - I||_F = {V_err.numpy():.4e}")
    
    # 2. Check Eigenvalues on real test data
    data_path = os.path.join(_SCRIPT_DIR, "fd003_conditioned_data.npz")
    loaded_data = np.load(data_path)
    r_test = loaded_data["R_test"].astype(np.float32)
    
    # Sample 10 r vectors
    np.random.seed(42)
    sample_indices = np.random.choice(r_test.shape[0], size=10, replace=False)
    r_samples = r_test[sample_indices]
    
    print("\nEigenvalue maximum absolute values for 10 samples:")
    
    # _get_K now takes batched r
    K_batched = model.koopman._get_K(tf.constant(r_samples))
    eigvals = tf.linalg.eigvals(tf.cast(K_batched, tf.complex64))
    max_eigs = tf.reduce_max(tf.abs(eigvals), axis=1)
    
    for i, meig in enumerate(max_eigs.numpy()):
        print(f"  Sample {i+1}: max(|λ|) = {meig:.6f} {'[STABLE]' if meig < 1.0 else '[UNSTABLE]'}")

if __name__ == "__main__":
    main()
