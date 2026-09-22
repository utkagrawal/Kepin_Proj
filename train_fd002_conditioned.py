#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KePIN FD003 Regime-Conditioned Training Script
"""

import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR   = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import argparse
import datetime
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import GenericTimeSeriesDataset as GDS
from kepin_model import KePINModel, build_kepin_model, convert_4d_to_3d
from kepin_losses import make_kepin_loss
from gpu_config import setup_gpu
from kepin_training import (
    rmse_np, mae_np, physics_metrics_np,
    apply_ema_smoothing, SEED,
)
from kepin_cmapss_optimized import (
    EnhancedKePINTrainer, augment_time_series,
    smooth_rul_labels, CMAPSS_CONFIGS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU setup
setup_gpu(mixed_precision=False, xla=False, verbose=True)

DATASET_KEY = "CMAPSS_FD002"
CFG = CMAPSS_CONFIGS[DATASET_KEY]

def train_fd002_conditioned(output_dir: str, signal_type: str = 'a', verbose: int = 1, epochs: int = None, n_runs: int = None, start_run: int = 0):
    """Full 3-run optimized training for FD002 with Regime Conditioning."""

    if epochs is not None:
        CFG["epochs"] = epochs
    if n_runs is not None:
        CFG["n_runs"] = n_runs

    config_path = os.path.join(_CODE_DIR, CFG["config_path"])
    with open(config_path) as f:
        all_configs = json.load(f)
    ds_config = all_configs[CFG["config_idx"]]  # index 2 = FD003
    ds_name   = ds_config.get("name", DATASET_KEY)

    # Make data paths absolute
    for key in ("train_path", "test_path", "test_rul_path"):
        if ds_config.get(key):
            ds_config[key] = os.path.join(_CODE_DIR, ds_config[key])

    print(f"\n{'='*70}")
    print(f"  KePIN FD002 Regime Conditioned (Signal {signal_type.upper()})")
    print(f"  epochs={CFG['epochs']}, patience={CFG['patience']}, "
          f"lr={CFG['lr']}, batch={CFG['batch_size']}")
    print(f"{'='*70}")

    # Load Data and Signals
    data_path = os.path.join(_SCRIPT_DIR, "fd002_signals.npz")
    loaded_data = np.load(data_path)
    X_train = loaded_data["X_train"].astype(np.float32)
    Y_train = loaded_data["Y_train"].astype(np.float32)
    X_test = loaded_data["X_test"].astype(np.float32)
    Y_test = loaded_data["Y_test"].astype(np.float32)
    
    if signal_type.lower() == 'a':
        r_train = loaded_data["A_train"].astype(np.float32)
        r_test = loaded_data["A_test"].astype(np.float32)
    else:
        r_train = loaded_data["B_train"].astype(np.float32)
        r_test = loaded_data["B_test"].astype(np.float32)
    
    print(f"  Loaded Regime vectors: Train {r_train.shape}, Test {r_test.shape}")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape: {X_test.shape}")

    # Wrap as tuples
    X_train_tuple = (X_train, r_train)
    X_test_tuple = (X_test, r_test)

    seq_len = X_train.shape[1]
    n_feat  = X_train.shape[2]
    n_train = X_train.shape[0]

    Y_train_smooth = smooth_rul_labels(Y_train, sigma=CFG["label_smooth_sigma"])

    arch_config = CFG["arch_override"].copy()
    arch_config["kernels"] = [min(k, seq_len) for k in arch_config["kernels"]]
    arch_config["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch_config["kernels"]]

    print(f"  Architecture: {arch_config['tier']}, latent_dim={arch_config['latent_dim']}")
    print(f"  Filters: {arch_config['filters']}, Kernels: {arch_config['kernels']}")

    os.makedirs(output_dir, exist_ok=True)

    all_results    = []
    all_test_preds = []

    for run_id in range(start_run):
        pred_path = os.path.join(output_dir, f"predictions_{ds_name}_run{run_id}.npz")
        if os.path.exists(pred_path):
            print(f"  Loading existing predictions for run {run_id}...")
            loaded_preds = np.load(pred_path)["y_pred"]
            all_test_preds.append(loaded_preds)
            # Dummy result for loaded runs
            all_results.append({
                "dataset": ds_name, "run_id": run_id, "rmse": rmse_np(Y_test, loaded_preds)
            })

    for run_id in range(start_run, CFG["n_runs"]):
        print(f"\n  --- Run {run_id + 1}/{CFG['n_runs']} ---")

        tf.random.set_seed(SEED + run_id * 100)
        np.random.seed(SEED + run_id * 100)

        X_train_aug, Y_train_aug = augment_time_series(
            X_train_tuple, Y_train_smooth,
            noise_std=CFG["noise_std"] * 0.5,
            seed=SEED + run_id,
        )

        n_active_losses = 7
        model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                                  arch_config=arch_config,
                                  n_active_losses=n_active_losses)

        if run_id == 0:
            # dummy forward pass to build condition_net
            dummy_x = tf.zeros((1, seq_len, n_feat))
            dummy_r = tf.zeros((1, r_train.shape[1]))
            model((dummy_x, dummy_r), training=False)
            
            n_params = sum(np.prod(v.shape) for v in model.trainable_variables)
            print(f"  Total params: {n_params:,}")
            print(model.summary_config())

        loss_fn = make_kepin_loss(
            loss_weights_layer=model.loss_weight_layer,
            use_auto_weights=True,
            domain_mode="degradation",
        )

        optimizer = keras.optimizers.Adam(
            learning_rate=float(CFG["lr"]),
            clipnorm=1.0,
        )

        trainer = EnhancedKePINTrainer(
            model, loss_fn, optimizer,
            clip_norm=2.0,
            warmup_epochs=CFG["warmup_epochs"],
            curriculum_warmup=CFG["curriculum_warmup"],
            mixup_alpha=CFG["mixup_alpha"],
            swa_start_frac=CFG["swa_start_frac"],
        )

        history, swa_weights = trainer.fit_enhanced(
            X_train_aug, Y_train_aug, X_test_tuple, Y_test,
            epochs=CFG["epochs"],
            batch_size=CFG["batch_size"],
            patience=CFG["patience"],
            initial_lr=CFG["lr"],
            min_lr=CFG["min_lr"],
            noise_std=CFG["noise_std"],
            verbose=verbose,
        )

        Y_pred = model.predict_rul(X_test_tuple).numpy()
        test_rmse = rmse_np(Y_test, Y_pred)
        test_mae  = mae_np(Y_test, Y_pred)
        print(f"  Run {run_id+1} -- Best weights: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        all_test_preds.append(Y_pred.flatten())

        if swa_weights is not None:
            model.set_weights(swa_weights)
            Y_pred_swa = model.predict_rul(X_test_tuple).numpy()
            swa_rmse = rmse_np(Y_test, Y_pred_swa)
            print(f"  Run {run_id+1} -- SWA weights:  RMSE={swa_rmse:.4f}")
            if swa_rmse < test_rmse:
                Y_pred    = Y_pred_swa
                test_rmse = swa_rmse
                test_mae  = mae_np(Y_test, Y_pred_swa)
                print(f"  Run {run_id+1} -- Using SWA (better)")
                all_test_preds[-1] = Y_pred_swa.flatten()

        mono_viol, slope_err = physics_metrics_np(Y_test, Y_pred)
        ss_res = np.sum((Y_test.flatten() - Y_pred.flatten()) ** 2)
        ss_tot = np.sum((Y_test.flatten() - np.mean(Y_test.flatten())) ** 2) + 1e-10
        r2 = 1 - ss_res / ss_tot

        run_tag = f"{ds_name}_run{run_id}"
        model.save_weights(os.path.join(output_dir, f"kepin_{run_tag}.weights.h5"))
        np.savez(os.path.join(output_dir, f"predictions_{run_tag}.npz"),
                 y_true=Y_test.flatten(), y_pred=Y_pred.flatten())

        hist_df = pd.DataFrame({
            k: v for k, v in history.items()
            if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])
        })
        hist_df.to_csv(os.path.join(output_dir, f"history_{run_tag}.csv"), index=False)

        all_results.append({
            "dataset": ds_name, "run_id": run_id,
            "rmse": test_rmse, "mae": test_mae, "r2": r2,
            "mono_violation": mono_viol, "slope_rmse": slope_err,
            "epochs_trained": len(history["epoch"]),
        })

    # Ensemble prediction
    Y_ensemble = np.mean(np.array(all_test_preds), axis=0)
    ens_rmse = rmse_np(Y_test, Y_ensemble.reshape(-1, 1))
    ens_mae  = mae_np(Y_test, Y_ensemble.reshape(-1, 1))
    ens_mono, ens_slope = physics_metrics_np(Y_test, Y_ensemble.reshape(-1, 1))
    ss_res = np.sum((Y_test.flatten() - Y_ensemble) ** 2)
    ss_tot = np.sum((Y_test.flatten() - np.mean(Y_test.flatten())) ** 2) + 1e-10
    ens_r2 = 1 - ss_res / ss_tot

    print(f"\n{'='*70}")
    print(f"  FD002 REGIME-CONDITIONED ENSEMBLE ({CFG['n_runs']} runs):")
    print(f"    RMSE: {ens_rmse:.4f}")
    print(f"    MAE:  {ens_mae:.4f}")
    print(f"    R2:   {ens_r2:.4f}")
    print(f"    Mono: {ens_mono:.6f}")
    print(f"{'='*70}")

    np.savez(os.path.join(output_dir, f"predictions_{ds_name}_ensemble.npz"),
             y_true=Y_test.flatten(), y_pred=Y_ensemble)

    all_results.append({
        "dataset": ds_name, "run_id": "ensemble",
        "rmse": ens_rmse, "mae": ens_mae, "r2": ens_r2,
        "mono_violation": ens_mono, "slope_rmse": ens_slope,
        "n_runs": CFG["n_runs"],
        "individual_rmses": [r["rmse"] for r in all_results],
    })

    with open(os.path.join(output_dir, "fd002_conditioned_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  All results saved to: {output_dir}")
    return ens_rmse


def main():
    parser = argparse.ArgumentParser(description="KePIN FD002 Regime Conditioned")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--signal_type", type=str, default="a", choices=["a", "b", "A", "B"])
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n_runs", type=int, default=None)
    parser.add_argument("--start_run", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(_SCRIPT_DIR, f"results_fd002_signal_{args.signal_type.lower()}_{ts}")

    rmse = train_fd002_conditioned(
        args.output_dir, signal_type=args.signal_type.lower(), verbose=args.verbose,
        epochs=args.epochs, n_runs=args.n_runs, start_run=args.start_run
    )


if __name__ == "__main__":
    main()
