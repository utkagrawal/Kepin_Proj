#!/usr/bin/env python3
"""
Run Regime-Aware KePIN Experiments on NASA C-MAPSS FD001.
"""

import os
import sys
import json
import time
import math
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from regime_kepin_model import (
    RegimeKePINModel, AutoBalancedLoss, compute_kepin_loss, auto_configure,
)

import GenericTimeSeriesDataset as GDS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(ds_config):
    ds = GDS.load_dataset_from_config(ds_config)
    X_train_4d, Y_train, X_test_4d, Y_test = ds.get_data()

    if X_train_4d.ndim == 4 and X_train_4d.shape[2] == 1:
        X_train = X_train_4d[:, :, 0, :]
        X_test = X_test_4d[:, :, 0, :]
    else:
        X_train = X_train_4d
        X_test = X_test_4d

    return X_train, Y_train.reshape(-1, 1), X_test, Y_test.reshape(-1, 1)

def make_dataloaders(X_train, Y_train, X_val, Y_val, batch_size=128):
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl

def rmse_np(y_true, y_pred):
    return float(np.sqrt(((y_true.flatten() - y_pred.flatten()) ** 2).mean()))

def mae_np(y_true, y_pred):
    return float(np.abs(y_true.flatten() - y_pred.flatten()).mean())

def nasa_score(y_true, y_pred):
    d = y_pred.flatten() - y_true.flatten()
    s = np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)
    return float(s.sum())

class CosineWarmRestarts:
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=1e-6, warmup=5):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.eta_min = eta_min
        self.warmup = warmup
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup:
            lr = self.base_lr * (epoch + 1) / self.warmup
        else:
            e = epoch - self.warmup
            T_cur = e % self.T_0
            lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * T_cur / self.T_0))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

def train_one_epoch(model, loss_balancer, optimizer, train_dl, clip_norm=2.0, aux_scale=1.0, 
                    no_physics=False, no_smooth=False):
    model.train()
    losses = []
    for X_b, Y_b in train_dl:
        X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
        optimizer.zero_grad()
        pred, kout = model(X_b)
        total, ld = compute_kepin_loss(Y_b, pred, kout, loss_balancer, aux_scale=aux_scale, 
                                       no_physics=no_physics, no_smooth=no_smooth)
        total.backward()
        nn.utils.clip_grad_norm_(list(model.parameters()), clip_norm)
        optimizer.step()
        losses.append(ld['total'])
    return np.mean(losses)

@torch.no_grad()
def evaluate(model, loss_balancer, val_dl, aux_scale=1.0, no_physics=False, no_smooth=False):
    model.eval()
    all_preds, all_targets, losses = [], [], []
    pi_seqs = []
    
    for X_b, Y_b in val_dl:
        X_b, Y_b = X_b.to(DEVICE), Y_b.to(DEVICE)
        pred, kout = model(X_b)
        total, ld = compute_kepin_loss(Y_b, pred, kout, loss_balancer, aux_scale=aux_scale, 
                                       no_physics=no_physics, no_smooth=no_smooth)
        losses.append(ld['total'])
        all_preds.append(pred.cpu().numpy())
        all_targets.append(Y_b.cpu().numpy())
        if kout['pi_seq'] is not None:
            pi_seqs.append(kout['pi_seq'].cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    pi_seq_all = np.concatenate(pi_seqs, axis=0) if pi_seqs else None
    
    return np.mean(losses), all_preds, all_targets, pi_seq_all

def run_experiment(ds_config, model_type, seed, output_dir):
    print(f"\n--- Running {model_type} (Seed {seed}) ---")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load Data
    X_train, Y_train_raw, X_test, Y_test_raw = load_dataset(ds_config)
    seq_len, n_feat = X_train.shape[1], X_train.shape[2]
    y_min, y_max = float(Y_train_raw.min()), float(Y_train_raw.max())
    y_range = max(y_max - y_min, 1e-6)
    Y_train = (Y_train_raw - y_min) / y_range
    Y_test = (Y_test_raw - y_min) / y_range

    # Hyperparams (from original tuning for FD001)
    epochs = 150 # Reduced from 300 to 150 for faster turnaround, since baseline used 150 in run_unified
    batch_size = 128
    lr = 0.0008
    patience = 40
    clip_norm = 2.0
    
    arch_config = auto_configure(n_feat, seq_len)
    model = RegimeKePINModel(seq_len, n_feat, arch_config, model_type=model_type).to(DEVICE)
    loss_balancer = AutoBalancedLoss(n_aux=4, aux_cap=0.5).to(DEVICE)
    
    print(model.summary_config())
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineWarmRestarts(optimizer, T_0=50, warmup=5, eta_min=1e-6)
    train_dl, val_dl = make_dataloaders(X_train, Y_train, X_test, Y_test, batch_size=batch_size)

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    no_physics = (model_type == 'proposed_nophysics')
    no_smooth = (model_type in ['proposed_nosmooth', 'proposed_nophysics', 'baseline', '3koopman'])

    start_time = time.time()
    for epoch in range(epochs):
        cur_lr = scheduler.step(epoch)
        aux_scale = min(1.0, max(0.0, (epoch - 10) / 30.0))

        train_loss = train_one_epoch(
            model, loss_balancer, optimizer, train_dl, clip_norm, 
            aux_scale=aux_scale, no_physics=no_physics, no_smooth=no_smooth
        )

        val_loss, val_preds_norm, val_tgts_norm, _ = evaluate(
            model, loss_balancer, val_dl, aux_scale=aux_scale, 
            no_physics=no_physics, no_smooth=no_smooth
        )

        val_rmse_orig = rmse_np(val_tgts_norm, val_preds_norm) * y_range

        if val_rmse_orig < best_val_rmse:
            best_val_rmse = val_rmse_orig
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val RMSE: {val_rmse_orig:.2f} | Pat: {patience_counter}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    _, test_preds_norm, test_tgts_norm, pi_seq_all = evaluate(
        model, loss_balancer, val_dl, no_physics=no_physics, no_smooth=no_smooth
    )

    test_preds = test_preds_norm * y_range + y_min
    test_tgts = test_tgts_norm * y_range + y_min
    
    test_rmse = rmse_np(test_tgts, test_preds)
    test_mae = mae_np(test_tgts, test_preds)
    test_nasa = nasa_score(test_tgts, test_preds)

    print(f"\nFinal Results - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, NASA: {test_nasa:.4f}")

    res_dir = os.path.join(output_dir, f"{model_type}_seed{seed}")
    os.makedirs(res_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(res_dir, "model.pt"))
    np.savez(os.path.join(res_dir, "predictions.npz"), y_true=test_tgts, y_pred=test_preds)
    
    if pi_seq_all is not None:
        np.save(os.path.join(res_dir, "pi_seq.npy"), pi_seq_all)
        
    return {
        "model": model_type,
        "seed": seed,
        "rmse": test_rmse,
        "mae": test_mae,
        "nasa": test_nasa,
        "params": model.count_params()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="datasets_cmapss_config.json")
    parser.add_argument("--output", default="experiments_regime")
    args = parser.parse_args()

    with open(args.config) as f:
        all_configs = json.load(f)
    
    fd001_config = next(c for c in all_configs if c["name"] == "CMAPSS_FD001")
    
    models = ["3koopman", "proposed", "proposed_nosmooth", "proposed_nophysics"]
    seeds = [42, 43, 44]
    
    results = []
    for m in models:
        for s in seeds:
            res = run_experiment(fd001_config, m, s, args.output)
            results.append(res)
            
    df = pd.DataFrame(results)
    
    print("\n\n" + "="*50)
    print("FINAL AGGREGATED RESULTS")
    print("="*50)
    agg = df.groupby("model")[["rmse", "mae", "nasa"]].agg(['mean', 'std']).round(4)
    print(agg)
    
    df.to_csv(os.path.join(args.output, "summary_results.csv"), index=False)
    agg.to_csv(os.path.join(args.output, "aggregated_results.csv"))

if __name__ == "__main__":
    main()
