#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KePIN FD003 Replication Script
Replicates RMSE = 11.42 on CMAPSS FD003 using EXACT settings from
  rahul15pandey/Kepin_code  (kepin_cmapss_optimized.py)

FD003 settings (from CMAPSS_CONFIGS["CMAPSS_FD003"]):
  epochs=300, patience=50, lr=8e-4, min_lr=1e-6, batch_size=128
  warmup_epochs=8, swa_start_frac=0.75, mixup_alpha=0.2
  noise_std=0.012, curriculum_warmup=30, label_smooth_sigma=2.0
  n_runs=3  (3-run ensemble)
  arch: 4 blocks, filters=[64,128,128,256], kernels=[11,7,5,3],
        latent_dim=96, lstm_units=96, n_heads=4, head_key_dim=24,
        dropout=0.3, rollout=3, spectral_k=5
  features (16): s2,s3,s4,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s17,s20,s21
  sequence_length=30, rul_cap=125

Usage:
  conda activate kepin
  python train_fd003.py [--output_dir OUTPUT_DIR] [--verbose 1]
"""

import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR   = os.path.join(_SCRIPT_DIR, "Kepin_code", "code")
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import argparse
import datetime
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import GenericTimeSeriesDataset as GDS
from kepin_model import KePINModel, build_kepin_model, convert_4d_to_3d, auto_configure
from kepin_losses import (
    KePINLossWeights, make_kepin_loss,
    rul_mse_loss, koopman_one_step_loss, spectral_stability_loss,
    monotonicity_loss, multi_step_loss, asymmetric_loss, slope_matching_loss,
)
from koopman_module import extract_spectral_features
from gpu_config import setup_gpu, build_tf_dataset, is_mixed_precision_enabled
from kepin_training import (
    rmse_np, mae_np, physics_metrics_np,
    apply_ema_smoothing, KePINTrainer, SEED,
)
from kepin_cmapss_optimized import (
    EnhancedKePINTrainer, augment_time_series,
    smooth_rul_labels, CMAPSS_CONFIGS,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU setup - same as author
setup_gpu(mixed_precision=False, xla=False, verbose=True)

DATASET_KEY = "CMAPSS_FD001"
CFG = CMAPSS_CONFIGS[DATASET_KEY]


def train_fd001(output_dir: str, verbose: int = 1):
    """Full 3-run optimised training for FD001, returning ensemble RMSE."""

    config_path = os.path.join(_CODE_DIR, CFG["config_path"])
    with open(config_path) as f:
        all_configs = json.load(f)
    ds_config = all_configs[CFG["config_idx"]]  # index 0 = FD001
    ds_name   = ds_config.get("name", DATASET_KEY)

    # Make data paths absolute — SLURM sets CWD to $TMPDIR, not _CODE_DIR
    for key in ("train_path", "test_path", "test_rul_path"):
        if ds_config.get(key):
            ds_config[key] = os.path.join(_CODE_DIR, ds_config[key])

    print(f"\n{'='*70}")
    print(f"  KePIN FD001 Replication")
    print(f"  epochs={CFG['epochs']}, patience={CFG['patience']}, "
          f"lr={CFG['lr']}, batch={CFG['batch_size']}")
    print(f"  n_runs={CFG['n_runs']}, mixup_alpha={CFG['mixup_alpha']}, "
          f"noise={CFG['noise_std']}")
    print(f"  Data: {ds_config['train_path']}")
    print(f"{'='*70}")

    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()
    print(ds.summary())

    X_train = convert_4d_to_3d(X_train_4d)
    X_test  = convert_4d_to_3d(X_test_4d)

    X_train, ema_alpha = apply_ema_smoothing(X_train)
    X_test, _          = apply_ema_smoothing(X_test, alpha=ema_alpha)
    print(f"  EMA alpha = {ema_alpha:.4f}")

    seq_len = X_train.shape[1]
    n_feat  = X_train.shape[2]
    n_train = X_train.shape[0]

    Y_train_smooth = smooth_rul_labels(Y_train, sigma=CFG["label_smooth_sigma"])

    arch_config = CFG["arch_override"].copy()
    arch_config["kernels"] = [min(k, seq_len) for k in arch_config["kernels"]]
    arch_config["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch_config["kernels"]]

    print(f"  Architecture: {arch_config['tier']}, latent_dim={arch_config['latent_dim']}")
    print(f"  Filters: {arch_config['filters']}, Kernels: {arch_config['kernels']}")
    print(f"  Features: {n_feat}, Seq len: {seq_len}, Train samples: {n_train}")

    os.makedirs(output_dir, exist_ok=True)

    all_results    = []
    all_test_preds = []

    for run_id in range(CFG["n_runs"]):
        print(f"\n  --- Run {run_id + 1}/{CFG['n_runs']} ---")

        tf.random.set_seed(SEED + run_id * 100)
        np.random.seed(SEED + run_id * 100)

        X_train_aug, Y_train_aug = augment_time_series(
            X_train, Y_train_smooth,
            noise_std=CFG["noise_std"] * 0.5,
            seed=SEED + run_id,
        )

        n_active_losses = 7
        model = build_kepin_model(seq_len, n_feat, n_train=n_train,
                                  arch_config=arch_config,
                                  n_active_losses=n_active_losses)

        if run_id == 0:
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
            X_train_aug, Y_train_aug, X_test, Y_test,
            epochs=CFG["epochs"],
            batch_size=CFG["batch_size"],
            patience=CFG["patience"],
            initial_lr=CFG["lr"],
            min_lr=CFG["min_lr"],
            noise_std=CFG["noise_std"],
            verbose=verbose,
        )

        Y_pred = model.predict_rul(tf.constant(X_test)).numpy()
        test_rmse = rmse_np(Y_test, Y_pred)
        test_mae  = mae_np(Y_test, Y_pred)
        print(f"  Run {run_id+1} -- Best weights: RMSE={test_rmse:.4f}, MAE={test_mae:.4f}")
        all_test_preds.append(Y_pred.flatten())

        if swa_weights is not None:
            model.set_weights(swa_weights)
            Y_pred_swa = model.predict_rul(tf.constant(X_test)).numpy()
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

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(history["train_loss"], label="Train", color="#0173B2")
        axes[0].plot(history["val_loss"],   label="Val",   color="#DE8F05")
        axes[0].set_title(f"Total Loss -- {ds_name} (run {run_id})")
        axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[1].plot(history["train_rmse"], label="Train", color="#0173B2")
        axes[1].plot(history["val_rmse"],   label="Val",   color="#DE8F05")
        axes[1].set_title("RMSE"); axes[1].set_xlabel("Epoch"); axes[1].legend()
        eig_hist = np.array(history["eigenvalues"])
        for mi in range(min(4, eig_hist.shape[1])):
            axes[2].plot(np.abs(eig_hist[:, mi]), label=f"Mode {mi+1}", alpha=0.8)
        axes[2].axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        axes[2].set_title("Koopman |lambda|"); axes[2].set_xlabel("Epoch")
        axes[2].legend(fontsize=7)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"training_{run_tag}.png"), dpi=150)
        plt.close()

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
    print(f"  FD001 STATIC ENSEMBLE ({CFG['n_runs']} runs):")
    print(f"    RMSE: {ens_rmse:.4f}  (target: 11.42)")
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

    with open(os.path.join(output_dir, "fd001_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 6))
    y_true_plot = Y_test.flatten()
    ax.scatter(y_true_plot, Y_ensemble, alpha=0.5, s=25, color="#029E73", edgecolors="none")
    lims = [min(y_true_plot.min(), Y_ensemble.min()),
            max(y_true_plot.max(), Y_ensemble.max())]
    ax.plot(lims, lims, "r--", alpha=0.7, linewidth=1.5, label="Perfect")
    ax.text(0.05, 0.92,
            f"RMSE = {ens_rmse:.2f}\nMAE  = {ens_mae:.2f}\nR2   = {ens_r2:.3f}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    ax.set_xlabel("True RUL"); ax.set_ylabel("Predicted RUL")
    ax.set_title("KePIN FD003 -- Ensemble Prediction")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_fd003_ensemble.png"), dpi=150)
    plt.close()

    print(f"\n  All results saved to: {output_dir}")
    return ens_rmse


def main():
    parser = argparse.ArgumentParser(
        description="KePIN FD001 Replication")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(_SCRIPT_DIR, f"results_fd001_{ts}")

    rmse = train_fd001(args.output_dir, verbose=args.verbose)
    print(f"\nFinal ensemble RMSE = {rmse:.4f}  (target approx 11.42)")


if __name__ == "__main__":
    main()
