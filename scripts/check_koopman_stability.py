import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from kepin_model import build_kepin_model, convert_4d_to_3d
import GenericTimeSeriesDataset as GDS
from kepin_cmapss_optimized import CMAPSS_CONFIGS

def main():
    print("=== Koopman Stability Diagnostic ===")
    
    # Load dataset to get correct dimensions and some real test samples
    print("Loading FD002 dataset...")
    cfg = CMAPSS_CONFIGS["CMAPSS_FD002"]
    
    config_path = os.path.join(os.path.dirname(__file__), '..', cfg["config_path"])
    with open(config_path) as f:
        all_configs = json.load(f)
    ds_config = all_configs[cfg["config_idx"]]
    
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    
    X_train = convert_4d_to_3d(X_train_4d)
    X_test = convert_4d_to_3d(X_test_4d)
    
    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]
    print(f"Data shapes: X_test: {X_test.shape}")
    
    # Build model (condition_dim=3, MLP condition net)
    print("Building model...")
    model = build_kepin_model(
        seq_len=seq_len,
        n_features=n_features,
        arch_config=cfg.get("arch_override", None),
        condition_dim=3,
        conditioning_strategy="raw3",
        condition_net_type="mlp",
        condition_indices=[0, 1, 2]
    )
    
    model.build(input_shape=(None, seq_len, n_features))
    
    # Load weights from the baseline FD002 run 0
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "experiments_result/phase11/fd002_cayley_exact/CMAPSS_FD002/kepin_CMAPSS_FD002_run0.weights.h5")
        
    print(f"Loading weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)
    
    # Extract U and V
    U = model.koopman.U.numpy()
    V = model.koopman.V.numpy()
    
    # Calculate Frobenius norms of (U^T U - I) and (V^T V - I)
    d = U.shape[0]
    I = np.eye(d)
    
    U_orth_error = np.linalg.norm(U.T @ U - I, ord='fro')
    V_orth_error = np.linalg.norm(V.T @ V - I, ord='fro')
    
    print("\n--- Orthogonality Drift ---")
    print(f"||U^T U - I||_F : {U_orth_error:.6f}")
    print(f"||V^T V - I||_F : {V_orth_error:.6f}")
    
    # Take ~100 real test samples
    sample_size = min(100, X_test.shape[0])
    X_sample = X_test[:sample_size]
    
    print(f"\n--- Spectral Stability Test (n={sample_size} samples) ---")
    
    # Directly use model on X_sample
    print("Running forward pass...")
    out = model(X_sample, training=False)
    
    print("Computing K batch manually...")
    mu = model.koopman.strategy_layer(X_sample)
    delta_s = model.koopman.condition_net(mu)
    sigma = tf.nn.sigmoid(model.koopman.s + delta_s)
    
    U_t = tf.cast(model.koopman.U, tf.complex64)
    V_t = tf.cast(model.koopman.V, tf.complex64)
    sigma_c = tf.cast(sigma, tf.complex64) # (batch, d)
    
    d = U_t.shape[0]
    batch_size = tf.shape(sigma_c)[0]
    U_b = tf.broadcast_to(tf.expand_dims(U_t, 0), [batch_size, d, d])
    V_b = tf.broadcast_to(tf.expand_dims(V_t, 0), [batch_size, d, d])
    
    S_b = tf.linalg.diag(sigma_c)
    K_batch = tf.matmul(U_b, tf.matmul(S_b, V_b, adjoint_b=True))
    K_batch = K_batch.numpy()
    
    sigmas_batch = sigma.numpy()
    
    max_eigs = []
    max_sigmas = []
    
    for i in range(sample_size):
        K = K_batch[i]
        sig = sigmas_batch[i]
        
        # True eigenvalues of K
        eigs = np.linalg.eigvals(K)
        max_eig = np.max(np.abs(eigs))
        
        # Proxy sigma
        max_sig = np.max(sig)
        
        max_eigs.append(max_eig)
        max_sigmas.append(max_sig)
        
    max_eigs = np.array(max_eigs)
    max_sigmas = np.array(max_sigmas)
    
    discrepancies = np.abs(max_eigs - max_sigmas)
    
    print(f"Average discrepancy between max(|eig|) and max(sigma): {np.mean(discrepancies):.6f}")
    print(f"Max discrepancy between max(|eig|) and max(sigma):     {np.max(discrepancies):.6f}")
    
    print("\nAverage max(|eig|): {:.6f}".format(np.mean(max_eigs)))
    print("Average max(sigma): {:.6f}".format(np.mean(max_sigmas)))
    
    if U_orth_error < 0.1 and V_orth_error < 0.1 and np.mean(discrepancies) < 0.05:
        print("\n=> RESULT: Stability proxy is mathematically sound. Orthogonality holds.")
    else:
        print("\n=> RESULT: Orthogonality drift detected! Need to add regularization.")

if __name__ == "__main__":
    main()
