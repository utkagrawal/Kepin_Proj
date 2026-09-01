import sys
import re

def patch_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Update argparse in __main__
    # Search for: parser.add_argument("--condition_dim", type=int, default=0, help="Condition vector size (e.g. 3 for FD002)")
    arg_str = '    parser.add_argument("--condition_dim", type=int, default=0,\n                        help="Condition vector size (e.g. 3 for FD002)")'
    new_arg_str = arg_str + '\n    parser.add_argument("--conditioning", type=str, default="raw3",\n                        choices=["raw3", "raw2_ab", "raw2_ac", "raw2_bc", "regime_embed", "sensor_topk", "hybrid"],\n                        help="Conditioning strategy to use")'
    if arg_str in content:
        content = content.replace(arg_str, new_arg_str)
    
    # 2. Update train_cmapss_optimized call in __main__
    # Search for: results = train_cmapss_optimized(ds_key, ds_dir, condition_dim=args.condition_dim, verbose=args.verbose)
    call_str = 'results = train_cmapss_optimized(ds_key, ds_dir, condition_dim=args.condition_dim, verbose=args.verbose)'
    new_call_str = 'results = train_cmapss_optimized(ds_key, ds_dir, condition_dim=args.condition_dim, conditioning_strategy=args.conditioning, verbose=args.verbose)'
    if call_str in content:
        content = content.replace(call_str, new_call_str)

    # 3. Update train_cmapss_optimized signature
    sig_str = 'def train_cmapss_optimized(dataset_key: str, data_dir: str,\n                           condition_dim: int = 0, verbose: int = 1) -> List[dict]:'
    new_sig_str = 'def train_cmapss_optimized(dataset_key: str, data_dir: str,\n                           condition_dim: int = 0, conditioning_strategy: str = "raw3", verbose: int = 1) -> List[dict]:'
    if sig_str in content:
        content = content.replace(sig_str, new_sig_str)
        
    # 4. Update the logic for condition_indices and KMeans
    # Locate the condition_indices block
    cond_block = """    # Determine condition indices dynamically if condition_dim > 0
    condition_indices = None
    if condition_dim > 0:
        # Find columns that start with "setting" in feature_cols
        feature_cols = ds_config["feature_cols"]
        setting_cols = [i for i, col in enumerate(feature_cols) if "setting" in col]
        condition_indices = setting_cols[:condition_dim]
        print(f"  Parameterized Koopman ON: condition_dim={condition_dim}")
        print(f"  Condition indices mapped to features: {condition_indices}")"""
        
    new_cond_block = """    # Determine condition indices dynamically if condition_dim > 0
    condition_indices = None
    kmeans_centroids = None
    if condition_dim > 0:
        feature_cols = ds_config["feature_cols"]
        setting_cols = [i for i, col in enumerate(feature_cols) if "setting" in col]
        
        if conditioning_strategy == "raw3":
            condition_indices = setting_cols[:3]
        elif conditioning_strategy == "raw2_ab":
            condition_indices = [setting_cols[0], setting_cols[1]]
        elif conditioning_strategy == "raw2_ac":
            condition_indices = [setting_cols[0], setting_cols[2]]
        elif conditioning_strategy == "raw2_bc":
            condition_indices = [setting_cols[1], setting_cols[2]]
        elif conditioning_strategy == "sensor_topk":
            # s1, s18, s5, s6, s21
            target_sensors = ["s1", "s18", "s5", "s6", "s21"]
            condition_indices = [i for i, col in enumerate(feature_cols) if col in target_sensors][:condition_dim]
        elif conditioning_strategy in ["regime_embed", "hybrid"]:
            condition_indices = setting_cols[:3] # We need all 3 settings for the embedding
            if conditioning_strategy == "hybrid":
                condition_indices = [setting_cols[0], setting_cols[1]] # raw2_ab default for hybrid raw_indices
            
            # Fit or load KMeans
            import os
            import pickle
            from sklearn.cluster import KMeans
            kmeans_path = "experiments_result/ablation_conditioning/regime_kmeans_FD002.pkl"
            if os.path.exists(kmeans_path):
                with open(kmeans_path, "rb") as f:
                    kmeans_model = pickle.load(f)
                kmeans_centroids = kmeans_model.cluster_centers_
                print("  Loaded pre-fitted KMeans from artifact.")
            else:
                print("  Fitting KMeans on training set for regime embedding...")
                # Extract original settings from training data (before smoothing/normalization)
                # Since X_train here is already normalized, we should ideally use raw data. 
                # But wait, X_train is normalized! K-Means on normalized settings is perfectly fine, 
                # it's just scaled. Let's fit on the normalized X_train settings.
                # X_train shape: (n_train, seq_len, n_features)
                X_train_settings = X_train[:, :, setting_cols[:3]].mean(axis=1) # (n_train, 3)
                kmeans_model = KMeans(n_clusters=6, random_state=42, n_init=10)
                kmeans_model.fit(X_train_settings)
                kmeans_centroids = kmeans_model.cluster_centers_
                os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
                with open(kmeans_path, "wb") as f:
                    pickle.dump(kmeans_model, f)
                print("  Fitted and saved KMeans to artifact.")

        print(f"  Parameterized Koopman ON: strategy={conditioning_strategy}")
        print(f"  Condition indices mapped to features: {condition_indices}")"""
    
    if cond_block in content:
        content = content.replace(cond_block, new_cond_block)

    # 5. Pass parameters to build_kepin_model
    build_str = """        model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                                  arch_config=arch_config,
                                  n_active_losses=n_active_losses,
                                  condition_dim=condition_dim,
                                  condition_indices=condition_indices)"""
    new_build_str = """        model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                                  arch_config=arch_config,
                                  n_active_losses=n_active_losses,
                                  condition_dim=condition_dim,
                                  condition_indices=condition_indices,
                                  conditioning_strategy=conditioning_strategy,
                                  kmeans_centroids=kmeans_centroids)"""
    if build_str in content:
        content = content.replace(build_str, new_build_str)

    with open(file_path, 'w') as f:
        f.write(content)

patch_file("kepin_cmapss_optimized.py")
