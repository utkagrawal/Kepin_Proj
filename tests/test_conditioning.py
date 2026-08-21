import sys
import os
import tensorflow as tf
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kepin_model import build_kepin_model

def test_strategies_shapes_and_nans():
    seq_len = 30
    n_feat = 24
    batch_size = 8
    
    # Dummy data
    np.random.seed(42)
    tf.random.set_seed(42)
    dummy_x = np.random.normal(size=(batch_size, seq_len, n_feat)).astype(np.float32)
    
    arch_config = {
        "latent_dim": 64,
        "rollout": 3,
        "spectral_k": 4,
        "dropout": 0.1,
        "filters": [16, 32, 64],
        "kernels": [3, 3, 3],
        "tier": "test"
    }
    
    strategies = [
        ("raw3", [0, 1, 2]),
        ("raw2_ab", [0, 1]),
        ("raw2_ac", [0, 2]),
        ("raw2_bc", [1, 2]),
        ("sensor_topk", [3, 4, 5, 6, 7]), # dummy indices
        ("regime_embed", [0, 1, 2]),
        ("hybrid", [0, 1]) # raw_indices
    ]
    
    # Fake centroids
    kmeans_centroids = np.random.normal(size=(6, 3)).astype(np.float32)
    
    for strategy_name, indices in strategies:
        tf.keras.backend.clear_session()
        print(f"Testing {strategy_name}...")
        
        model = build_kepin_model(
            seq_len=seq_len, 
            n_features=n_feat, 
            arch_config=None,
            condition_dim=len(indices) if strategy_name != "hybrid" else 3,
            condition_indices=indices,
            conditioning_strategy=strategy_name,
            kmeans_centroids=kmeans_centroids if strategy_name in ["regime_embed", "hybrid"] else None
        )
        
        # Build model by passing dummy data
        out, _ = model(dummy_x, training=False)
        
        # Check shapes
        assert out.shape == (batch_size, 1), f"Expected (batch, 1) output shape for {strategy_name}"
        
        # Check NaNs
        assert not np.any(np.isnan(out.numpy())), f"NaN detected in output for {strategy_name}"
        
        print(f"  {strategy_name} OK")

if __name__ == "__main__":
    test_strategies_shapes_and_nans()
    print("All strategy shape and NaN tests passed.")
