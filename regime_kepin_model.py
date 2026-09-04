#!/usr/bin/env python3
"""
Regime-Aware KePIN PyTorch Model.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegimeKoopmanOperator(nn.Module):
    def __init__(self, latent_dim, rollout_steps=3, sigma_max=0.99, num_regimes=3, 
                 use_regimes=True, static_regimes=False, no_physics=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.sigma_max = sigma_max
        self.num_regimes = num_regimes if use_regimes else 1
        self.use_regimes = use_regimes
        self.static_regimes = static_regimes
        self.no_physics = no_physics

        self.U_raw = nn.Parameter(torch.randn(self.num_regimes, latent_dim, latent_dim) * 0.01)
        self.V_raw = nn.Parameter(torch.randn(self.num_regimes, latent_dim, latent_dim) * 0.01)
        self.sigma_raw = nn.Parameter(torch.zeros(self.num_regimes, latent_dim))

        if use_regimes:
            if static_regimes:
                self.static_pi_logits = nn.Parameter(torch.zeros(num_regimes))
            else:
                self.regime_net = nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_regimes)
                )

    def _orthogonalize(self, M):
        Q, R = torch.linalg.qr(M)
        return Q

    def _get_K_all(self):
        U = self._orthogonalize(self.U_raw)
        V = self._orthogonalize(self.V_raw)
        sigma = self.sigma_max * torch.sigmoid(self.sigma_raw)
        K = U @ torch.diag_embed(sigma) @ V.transpose(-2, -1) # (R, d, d)
        return K

    def compute_pi(self, z):
        """z: (..., d)"""
        if not self.use_regimes:
            return torch.ones(*z.shape[:-1], 1, device=z.device)
            
        if self.static_regimes:
            pi = F.softmax(self.static_pi_logits, dim=-1)
            return pi.expand(*z.shape[:-1], -1)
        else:
            return F.softmax(self.regime_net(z), dim=-1)

    def forward(self, z_seq):
        """
        Args:
            z_seq: (batch, T, d) latent states
        """
        K_all = self._get_K_all() # (R, d, d)
        batch, T, d = z_seq.shape

        pi_seq = self.compute_pi(z_seq) # (B, T, R)
        
        if not self.use_regimes:
            K = K_all[0]
            z_t = z_seq[:, :-1, :]
            z_tp1 = z_seq[:, 1:, :]
            one_step_pred = torch.einsum('ij,btj->bti', K, z_t)
            
            horizons = list(range(2, min(self.rollout_steps + 2, T)))
            multi_preds, multi_targets = [], []
            K_pow = K.clone()
            for h in horizons:
                K_pow = K_pow @ K if h > 2 else K @ K
                n_valid = T - h
                if n_valid <= 0: break
                multi_preds.append(torch.einsum('ij,btj->bti', K_pow, z_seq[:, :n_valid, :]))
                multi_targets.append(z_seq[:, h:h+n_valid, :])
            
            eigenvalues = torch.linalg.eigvals(K).unsqueeze(0).expand(batch, -1) # (B, d)
        else:
            z_t = z_seq[:, :-1, :]
            pi_t = pi_seq[:, :-1, :] # (B, T-1, R)
            K_t = torch.einsum('btr,rij->btij', pi_t, K_all) # (B, T-1, d, d)
            one_step_pred = torch.einsum('btij,btj->bti', K_t, z_t)
            z_tp1 = z_seq[:, 1:, :]
            
            horizons = list(range(2, min(self.rollout_steps + 2, T)))
            multi_preds, multi_targets = [], []
            
            if len(horizons) > 0:
                z_curr = one_step_pred 
                for h in horizons:
                    n_valid = T - h
                    if n_valid <= 0: break
                    
                    pi_curr = self.compute_pi(z_curr[:, :n_valid, :]) # (B, n_valid, R)
                    K_curr = torch.einsum('btr,rij->btij', pi_curr, K_all)
                    z_next = torch.einsum('btij,btj->bti', K_curr, z_curr[:, :n_valid, :])
                    
                    multi_preds.append(z_next)
                    multi_targets.append(z_seq[:, h:h+n_valid, :])
                    z_curr = z_next 

            eigenvalues = torch.linalg.eigvals(K_all).view(-1) # (R*d,)
            eigenvalues = eigenvalues.unsqueeze(0).expand(batch, -1)

        if multi_preds:
            min_t = min(p.shape[1] for p in multi_preds)
            multi_pred = torch.stack([p[:, :min_t, :] for p in multi_preds], dim=2)
            multi_tgt = torch.stack([t[:, :min_t, :] for t in multi_targets], dim=2)
        else:
            multi_pred = one_step_pred[:, :1, :].unsqueeze(2)
            multi_tgt = z_tp1[:, :1, :].unsqueeze(2)

        return {
            'one_step_pred': one_step_pred,
            'one_step_target': z_tp1,
            'multi_step_pred': multi_pred,
            'multi_step_target': multi_tgt,
            'eigenvalues': eigenvalues,
            'K_all': K_all,
            'pi_seq': pi_seq
        }

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        s = x.mean(dim=-1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)

class ResConv1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.skip_bn = nn.BatchNorm1d(out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip_bn(self.skip(x)) if isinstance(self.skip, nn.Conv1d) else x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.relu(h + residual)

def auto_configure(n_features, seq_len):
    if n_features <= 10: tier = "small"
    elif n_features <= 16: tier = "medium"
    else: tier = "large"

    configs = {
        "small": {"filters": [64, 128, 128], "kernels": [7, 5, 3], "latent_dim": 64, "lstm_units": 64, "n_heads": 4, "dropout": 0.3, "rollout": 3},
        "medium": {"filters": [64, 128, 128, 256], "kernels": [7, 5, 5, 3], "latent_dim": 128, "lstm_units": 128, "n_heads": 4, "dropout": 0.35, "rollout": 3},
        "large": {"filters": [64, 128, 256, 256], "kernels": [7, 5, 5, 3], "latent_dim": 128, "lstm_units": 128, "n_heads": 8, "dropout": 0.4, "rollout": 3},
    }
    cfg = configs[tier]
    cfg["tier"] = tier
    cfg["kernels"] = [min(k, seq_len) | 1 for k in cfg["kernels"]]
    return cfg

class RegimeKePINModel(nn.Module):
    def __init__(self, seq_len, n_features, arch_config=None, model_type='proposed'):
        super().__init__()
        if arch_config is None:
            arch_config = auto_configure(n_features, seq_len)
        self.arch_config = arch_config
        self.model_type = model_type

        filters = arch_config["filters"]
        kernels = arch_config["kernels"]
        latent_dim = arch_config["latent_dim"]
        lstm_units = arch_config["lstm_units"]
        n_heads = arch_config["n_heads"]
        dropout = arch_config["dropout"]

        self.input_proj = nn.Conv1d(n_features, filters[0], 1)
        self.input_bn = nn.BatchNorm1d(filters[0])

        self.encoder = nn.ModuleList()
        in_ch = filters[0]
        for i, (f, k) in enumerate(zip(filters, kernels)):
            self.encoder.append(ResConv1DBlock(in_ch, f, k))
            in_ch = f

        self.bilstm = nn.LSTM(filters[-1], lstm_units, batch_first=True, bidirectional=True, dropout=dropout * 0.5)
        self.lstm_ln = nn.LayerNorm(lstm_units * 2)
        self.post_lstm = nn.Conv1d(lstm_units * 2, latent_dim, 1)
        self.post_lstm_bn = nn.BatchNorm1d(latent_dim)

        self.mha = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout * 0.3, batch_first=True)
        self.mha_ln = nn.LayerNorm(latent_dim)
        self.ff = nn.Sequential(nn.Linear(latent_dim, latent_dim * 2), nn.GELU(), nn.Linear(latent_dim * 2, latent_dim), nn.Dropout(dropout * 0.5))
        self.ff_ln = nn.LayerNorm(latent_dim)

        self.latent_proj = nn.Conv1d(latent_dim, latent_dim, 1)
        self.latent_bn = nn.BatchNorm1d(latent_dim)

        use_regimes = model_type in ['proposed', 'proposed_nosmooth', 'proposed_nophysics', '3koopman']
        static_regimes = model_type == '3koopman'
        no_physics = model_type == 'proposed_nophysics'
        
        self.koopman = RegimeKoopmanOperator(
            latent_dim, rollout_steps=arch_config["rollout"],
            use_regimes=use_regimes, static_regimes=static_regimes, no_physics=no_physics
        )

        head_in = latent_dim * 2 + latent_dim 
        self.head = nn.Sequential(
            nn.Linear(head_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head_skip = nn.Linear(latent_dim * 2, 64)
        self.head_out = nn.Linear(64, 1)

    def forward(self, x):
        B = x.shape[0]

        h = x.permute(0, 2, 1)
        h = F.relu(self.input_bn(self.input_proj(h)))
        for block in self.encoder: h = block(h)

        h = h.permute(0, 2, 1)
        h, _ = self.bilstm(h)
        h = self.lstm_ln(h)
        h = h.permute(0, 2, 1)
        h = F.relu(self.post_lstm_bn(self.post_lstm(h)))

        h = h.permute(0, 2, 1)
        attn_out, _ = self.mha(h, h, h)
        h = self.mha_ln(h + attn_out)
        ff_out = self.ff(h)
        h = self.ff_ln(h + ff_out)

        z = h.permute(0, 2, 1)
        z = self.latent_bn(self.latent_proj(z))
        z = z.permute(0, 2, 1)  # (B, T, d)

        koopman_out = self.koopman(z)

        pool_avg = z.mean(dim=1)
        pool_max = z.max(dim=1)[0]
        pooled = torch.cat([pool_avg, pool_max], dim=-1)

        eigs = koopman_out['eigenvalues'] 
        spec_feats = torch.cat([torch.abs(eigs[:, :self.arch_config["latent_dim"]])], dim=-1)

        head_in = torch.cat([pooled, spec_feats], dim=-1)
        h_deep = self.head(head_in)
        h_skip = F.relu(self.head_skip(pooled))
        pred = self.head_out(h_deep + h_skip)

        return pred, koopman_out

    def predict(self, x):
        self.eval()
        with torch.no_grad(): pred, _ = self(x)
        return pred

    def get_eigenvalues(self):
        K_all = self.koopman._get_K_all()
        return torch.linalg.eigvals(K_all).detach().cpu().numpy()

    def get_koopman_matrix(self):
        return self.koopman._get_K_all().detach().cpu().numpy()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def summary_config(self):
        cfg = self.arch_config
        n = self.count_params()
        return (f"RegimeKePIN({self.model_type}) | params={n:,} | "
                f"latent={cfg['latent_dim']} | "
                f"blocks={len(cfg['filters'])} | "
                f"heads={cfg['n_heads']}")

def prediction_loss(y_true, y_pred, delta=0.15):
    return F.huber_loss(y_pred.flatten(), y_true.flatten(), reduction='mean', delta=delta)

def koopman_consistency_loss(one_step_pred, one_step_target):
    return F.mse_loss(one_step_pred, one_step_target)

def spectral_stability_loss(eigenvalues):
    mags = torch.abs(eigenvalues)
    violation = F.relu(mags - 1.0)
    return (violation ** 2).mean()

def multistep_rollout_loss(multi_pred, multi_target):
    return F.mse_loss(multi_pred, multi_target)
    
def regime_smoothness_loss(pi_seq):
    if pi_seq is None or pi_seq.shape[-1] == 1:
        return torch.tensor(0.0, device=pi_seq.device if pi_seq is not None else 'cpu')
    pi_diff = pi_seq[:, 1:, :] - pi_seq[:, :-1, :]
    return (pi_diff ** 2).mean()

class AutoBalancedLoss(nn.Module):
    def __init__(self, n_aux=4, aux_cap=0.5):
        super().__init__()
        self.aux_weight = aux_cap
        self.n_aux = n_aux
        self.log_vars = nn.Parameter(torch.zeros(n_aux), requires_grad=False)

    def forward(self, losses):
        pred_loss = losses[0]
        aux_losses = losses[1:]
        total = pred_loss
        for aux_l in aux_losses:
            total = total + self.aux_weight * aux_l
        all_weights = torch.tensor([1.0] + [self.aux_weight] * self.n_aux, device=pred_loss.device)
        return total, all_weights

def compute_kepin_loss(y_true, y_pred, koopman_out, loss_balancer, aux_scale=1.0, no_physics=False, no_smooth=False):
    l_pred = prediction_loss(y_true, y_pred)
    l_koop = koopman_consistency_loss(koopman_out['one_step_pred'], koopman_out['one_step_target'])
    l_spec = spectral_stability_loss(koopman_out['eigenvalues'])
    l_multi = multistep_rollout_loss(koopman_out['multi_step_pred'], koopman_out['multi_step_target'])
    l_smooth = regime_smoothness_loss(koopman_out['pi_seq'])

    if no_physics:
        l_koop = l_koop * 0.0
        l_spec = l_spec * 0.0
        l_multi = l_multi * 0.0
        
    if no_smooth:
        l_smooth = l_smooth * 0.0

    total, weights = loss_balancer([l_pred, l_koop * aux_scale, l_spec * aux_scale, l_multi * aux_scale, l_smooth * aux_scale])

    loss_dict = {
        'total': total.item(),
        'pred': l_pred.item(),
        'koopman': l_koop.item(),
        'spectral': l_spec.item(),
        'multistep': l_multi.item(),
        'smooth': l_smooth.item(),
        'weights': weights.cpu().numpy(),
    }
    return total, loss_dict
