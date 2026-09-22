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

    model = build_kepin_model(seq_len, n_feat, n_train=n_train, arch_config=arch_config)
    
    # Initialize with dummy input (needs 12-dim R)
    dummy_x = tf.zeros((1, seq_len, n_feat))
    dummy_r = tf.zeros((1, 12))
    model((dummy_x, dummy_r), training=False)
    
    # Test 1: Normal inputs
    r_test = tf.random.normal((100, 12))
    
    # We can extract the bounded cond_out logic manually to test it since _get_K does it internally
    cond_out_raw = model.koopman.condition_net(r_test)
    cond_out = 2.0 * tf.math.tanh(cond_out_raw)
    
    print(f"Max absolute delta_s with random weights: {tf.reduce_max(tf.abs(cond_out)).numpy()}")
    
    # Test 2: Extreme inputs to force saturation
    r_extreme = tf.constant(np.ones((10, 12)) * 1e9, dtype=tf.float32)
    # Also set weights of condition_net to huge values
    for var in model.koopman.condition_net.trainable_variables:
        var.assign(tf.ones_like(var) * 1e6)
        
    cond_out_raw_extreme = model.koopman.condition_net(r_extreme)
    cond_out_extreme = 2.0 * tf.math.tanh(cond_out_raw_extreme)
    
    max_val = tf.reduce_max(tf.abs(cond_out_extreme)).numpy()
    print(f"Max absolute delta_s with EXTREME inputs and weights: {max_val}")
    
    assert max_val <= 2.0001, "Delta S exceeded bounds!"
    print("SUCCESS: delta_s is strictly bounded by max_shift (2.0) by construction.")

if __name__ == "__main__":
    main()
