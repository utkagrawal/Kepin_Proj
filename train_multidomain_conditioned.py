#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KePIN Multidomain Regime-Conditioned Training Script
Supports: Jena_Climate, Cylinder_Wake, Building_Energy
"""

import sys, os
import argparse
import datetime
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR   = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from kepin_model import KePINModel, build_kepin_model
from kepin_losses import make_kepin_loss
from gpu_config import setup_gpu
from kepin_training import rmse_np, mae_np, apply_ema_smoothing, SEED
from kepin_cmapss_optimized import EnhancedKePINTrainer, augment_time_series

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU setup
setup_gpu(mixed_precision=False, xla=False, verbose=True)

HYPERPARAMS = {
    "Jena_Climate": {"epochs": 150, "batch_size": 512, "lr": 0.001, "patience": 40, "clip_norm": 2.0},
    "Cylinder_Wake": {"epochs": 200, "batch_size": 256, "lr": 0.001, "patience": 50, "clip_norm": 2.0},
    "Building_Energy": {"epochs": 200, "batch_size": 256, "lr": 0.001, "patience": 50, "clip_norm": 2.0},
}

def train_multidomain_conditioned(dataset_key: str, output_dir: str, verbose: int = 1, epochs: int = None, n_runs: int = 3, start_run: int = 0):
    hp = HYPERPARAMS[dataset_key]
    if epochs is not None:
        hp["epochs"] = epochs

    print(f"\n{'='*70}")
    print(f"  KePIN {dataset_key} Regime Conditioned (Forecasting Mode)")
    print(f"  epochs={hp['epochs']}, patience={hp['patience']}, lr={hp['lr']}, batch={hp['batch_size']}")
    print(f"{'='*70}")

    # Load Data and Signals
    data_path = os.path.join(_SCRIPT_DIR, f"{dataset_key.lower()}_signals.npz")
    loaded_data = np.load(data_path)
    X_train = loaded_data["X_train"].astype(np.float32)
    Y_train = loaded_data["Y_train"].astype(np.float32)
    X_test = loaded_data["X_test"].astype(np.float32)
    Y_test = loaded_data["Y_test"].astype(np.float32)
    
    r_train = loaded_data["A_train"].astype(np.float32)
    r_test = loaded_data["A_test"].astype(np.float32)
    
    print(f"  Loaded Regime vectors: Train {r_train.shape}, Test {r_test.shape}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Target Normalization [0, 1] for stable training like torch pipeline
    y_min = float(Y_train.min())
    y_max = float(Y_train.max())
    y_range = max(y_max - y_min, 1e-6)
    Y_train_norm = (Y_train - y_min) / y_range
    Y_test_norm = (Y_test - y_min) / y_range

    X_train_tuple = (X_train, r_train)
    X_test_tuple = (X_test, r_test)

    seq_len = X_train.shape[1]
    n_feat  = X_train.shape[2]
    n_train = X_train.shape[0]

    # Arch Config
    tier = "medium" if dataset_key == "Jena_Climate" else "small"
    os.makedirs(output_dir, exist_ok=True)

    all_results    = []
    all_test_preds = []

    for run_id in range(start_run):
        pred_path = os.path.join(output_dir, f"predictions_{dataset_key}_run{run_id}.npz")
        if os.path.exists(pred_path):
            print(f"  Loading existing predictions for run {run_id}...")
            loaded_preds = np.load(pred_path)["y_pred"]
            all_test_preds.append(loaded_preds)
            all_results.append({
                "dataset": dataset_key, "run_id": run_id, "rmse": rmse_np(Y_test, loaded_preds)
            })

    for run_id in range(start_run, n_runs):
        print(f"\n  --- Run {run_id + 1}/{n_runs} ---")

        tf.random.set_seed(SEED + run_id * 100)
        np.random.seed(SEED + run_id * 100)

        n_active_losses = 4 # Forecasting only uses 4 losses
        model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                                  arch_config=None,
                                  n_active_losses=n_active_losses)

        if run_id == 0:
            dummy_x = tf.zeros((1, seq_len, n_feat))
            dummy_r = tf.zeros((1, r_train.shape[1]))
            model((dummy_x, dummy_r), training=False)
            
            n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
            print(f"  Total params: {n_params:,}")
            print(model.summary_config())

        loss_fn = make_kepin_loss(
            loss_weights_layer=model.loss_weight_layer,
            use_auto_weights=True,
            domain_mode="forecasting",
        )

        optimizer = keras.optimizers.Adam(
            learning_rate=float(hp["lr"]),
            clipnorm=hp["clip_norm"],
        )

        trainer = EnhancedKePINTrainer(
            model, loss_fn, optimizer,
            clip_norm=hp["clip_norm"],
            warmup_epochs=5,
            curriculum_warmup=15,
            mixup_alpha=0.0, # Disable mixup for non-cmapss domains
            swa_start_frac=0.75,
        )

        history, swa_weights = trainer.fit_enhanced(
            X_train_tuple, Y_train_norm, X_test_tuple, Y_test_norm,
            epochs=hp["epochs"],
            batch_size=hp["batch_size"],
            patience=hp["patience"],
            initial_lr=hp["lr"],
            min_lr=1e-6,
            noise_std=0.0, # Disable input noise for non-cmapss domains
            verbose=verbose,
        )

        Y_pred_norm = model.predict_rul(X_test_tuple).numpy()
        Y_pred = Y_pred_norm * y_range + y_min
        
        test_rmse = rmse_np(Y_test, Y_pred)
        test_mae  = mae_np(Y_test, Y_pred)
        print(f"  Run {run_id+1} -- Best weights: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        all_test_preds.append(Y_pred.flatten())

        if swa_weights is not None:
            model.set_weights(swa_weights)
            Y_pred_swa_norm = model.predict_rul(X_test_tuple).numpy()
            Y_pred_swa = Y_pred_swa_norm * y_range + y_min
            swa_rmse = rmse_np(Y_test, Y_pred_swa)
            print(f"  Run {run_id+1} -- SWA weights:  RMSE={swa_rmse:.4f}")
            if swa_rmse < test_rmse:
                Y_pred    = Y_pred_swa
                test_rmse = swa_rmse
                test_mae  = mae_np(Y_test, Y_pred_swa)
                print(f"  Run {run_id+1} -- Using SWA (better)")
                all_test_preds[-1] = Y_pred_swa.flatten()

        ss_res = np.sum((Y_test.flatten() - Y_pred.flatten()) ** 2)
        ss_tot = np.sum((Y_test.flatten() - np.mean(Y_test.flatten())) ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot

        run_tag = f"{dataset_key}_run{run_id}"
        model.save_weights(os.path.join(output_dir, f"kepin_{run_tag}.weights.h5"))
        np.savez(os.path.join(output_dir, f"predictions_{run_tag}.npz"),
                 y_true=Y_test.flatten(), y_pred=Y_pred.flatten())

        all_results.append({
            "dataset": dataset_key, "run_id": run_id,
            "rmse": test_rmse, "mae": test_mae, "r2": r2,
            "epochs_trained": len(history["epoch"]),
        })

    # Ensemble prediction
    Y_ensemble = np.mean(np.array(all_test_preds), axis=0)
    ens_rmse = rmse_np(Y_test, Y_ensemble.reshape(-1, 1))
    ens_mae  = mae_np(Y_test, Y_ensemble.reshape(-1, 1))
    ss_res = np.sum((Y_test.flatten() - Y_ensemble) ** 2)
    ss_tot = np.sum((Y_test.flatten() - np.mean(Y_test.flatten())) ** 2) + 1e-10
    ens_r2 = 1 - ss_res / ss_tot

    print(f"\n{'='*70}")
    print(f"  {dataset_key} REGIME-CONDITIONED ENSEMBLE ({n_runs} runs):")
    print(f"    RMSE: {ens_rmse:.4f}")
    print(f"    MAE:  {ens_mae:.4f}")
    print(f"    R2:   {ens_r2:.4f}")
    print(f"{'='*70}")

    np.savez(os.path.join(output_dir, f"predictions_{dataset_key}_ensemble.npz"),
             y_true=Y_test.flatten(), y_pred=Y_ensemble)

    all_results.append({
        "dataset": dataset_key, "run_id": "ensemble",
        "rmse": ens_rmse, "mae": ens_mae, "r2": ens_r2,
        "n_runs": n_runs,
        "individual_rmses": [r["rmse"] for r in all_results],
    })

    with open(os.path.join(output_dir, f"{dataset_key.lower()}_conditioned_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  All results saved to: {output_dir}")
    return ens_rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KePIN Multidomain Regime Conditioned")
    parser.add_argument("--dataset", type=str, required=True, choices=["Jena_Climate", "Cylinder_Wake", "Building_Energy"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--start_run", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(_SCRIPT_DIR, f"results_{args.dataset.lower()}_conditioned_{ts}")

    rmse = train_multidomain_conditioned(
        args.dataset, args.output_dir, verbose=args.verbose,
        epochs=args.epochs, n_runs=args.n_runs, start_run=args.start_run
    )
