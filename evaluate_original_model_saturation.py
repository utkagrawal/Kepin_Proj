import os
import sys
import json
import numpy as np
import tensorflow as tf

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import build_kepin_model

def get_original_results_dir():
    # Job 2791 was the original unfixed run
    return os.path.join(_SCRIPT_DIR, "results_fd003_cond_2791")

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
    # EVALUATE ORIGINAL UNFIXED MODEL (3D)
    # ==========================================
    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    # Needs 3-dim R!
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, 3))
    model((dummy_x, dummy_r), training=False)

    results_dir = get_original_results_dir()
    print(f"Loading ORIGINAL weights from {results_dir}...")
    weights_path = os.path.join(results_dir, "kepin_CMAPSS_FD003_run0.weights.h5")
    model.load_weights(weights_path)

    # To get original saturation, we don't even need the exact R_test! 
    # But let's build the autoencoder and recreate the exact R_test.
    import pandas as pd
    from prepare_fd003_conditioned import build_autoencoder, extract_causal_features
    import GenericTimeSeriesDataset as GDS
    
    cfg = {"name": "CMAPSS_FD003", "train_path": "C-MAPSS-Data/train_FD003.txt", "test_path": "C-MAPSS-Data/test_FD003.txt"}
    cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(os.path.join(_CODE_DIR, cfg['train_path']), sep='\\s+', names=cols)
    test_df = pd.read_csv(os.path.join(_CODE_DIR, cfg['test_path']), sep='\\s+', names=cols)
    
    causal_cols = ['s2','s3','s4','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
    causal_test = extract_causal_features(test_df, 'unit', 'cycle', causal_cols, seq_len=30, last_only=True)
    causal_train = extract_causal_features(train_df, 'unit', 'cycle', causal_cols, seq_len=30, last_only=False)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    causal_train_scaled = scaler.fit_transform(causal_train)
    causal_test_scaled = scaler.transform(causal_test)
    
    tf.random.set_seed(42)
    np.random.seed(42)
    ae, encoder = build_autoencoder(input_dim=causal_train.shape[1], latent_dim=3)
    ae.fit(causal_train_scaled, causal_train_scaled, epochs=20, batch_size=256, verbose=0)
    
    R_test_orig = encoder.predict(causal_test_scaled, batch_size=512)
    
    s = model.koopman.s.numpy() # shape: (96,)
    
    # Original model used unconstrained cond_out
    cond_out = model.koopman.condition_net(tf.constant(R_test_orig)).numpy()
    
    sigma = tf.nn.sigmoid(s + cond_out).numpy()
    
    # 2. SATURATION CHECK
    healthy = np.sum((sigma >= 0.05) & (sigma <= 0.95))
    total = sigma.size
    fraction_healthy = healthy / total
    
    print("\n=========================================================")
    print(" ORIGINAL (UNFIXED) SATURATION CHECK")
    print("=========================================================")
    print(f"Sigma(r) Healthy [0.05, 0.95] Fraction: {fraction_healthy:.2%}")

if __name__ == "__main__":
    main()
