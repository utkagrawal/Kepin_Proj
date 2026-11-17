#!/usr/bin/env python3
"""
Train one KePIN variant on one C-MAPSS subset for one run (seed).

Arms
  --koopman svd --mode static   original KePIN Koopman layer (reference)
  --koopman eig --mode static   eigen-parameterised layer, fixed operator
  --koopman eig --mode exo      operator changes every step with the operating condition (variant A)
  --koopman eig --mode health   decay shifted by a learned per-window health index (variant B)
  --koopman eig --mode both     A + B
  --koopman eig --mode static --head ttf
                                Koopman time-to-failure: RUL = steps until the Koopman-projected
                                health crosses the failure threshold (Koopman *is* the predictor)
  --head wear                   TTF + wear-accumulator mode (eigenvalue 1, learned wear rate)
  --koopman none --mode static  ablation: Koopman removed completely (no layer, no spectral
                                features, no Koopman losses); same encoder and RUL head
  --future_loss W               extra Koopman loss: project today's latent h = 5..60 flights ahead
                                onto the encoder's latent of the same engine's REAL future window
  --future_subspace K           ...only on 2K dedicated latent dims, with a growth (eigenvalue-1) term
  --cond_norm                   learned per-condition input normalisation (before in-model EMA)
  --koop_control                Koopman with control: latent = condition-free state + C*condition
  --no_koopman_loss             ablation: same model, Koopman losses removed (does Koopman help?)

Regime-adaptive Koopman (the decay of every mode depends on the engine's health stage)
  --koopman eig --mode health --cond_norm --health_map ordered --head_state 10 --regime_loss X
      health h per window; h = 1 healthy (slow decay), h = 0 near failure (fast decay);
      the head also gets the state carried 10 flights ahead by K(h).
      X = none (h unconstrained) | mono (h must fall over each engine's life) | sup (h ~ RUL/cap)

Protocols
  paper : validation = test set (what the original KePIN code does: best epoch and
          SWA-vs-best are picked on test RMSE). Only for comparison with the paper table.
  clean : validation = 20% of training engines held out; the test set is touched once.

Training recipe, hyper-parameters and seeds are the original KePIN ones
(CMAPSS_CONFIGS, EnhancedKePINTrainer, seed = 42 + 100*run_id), so arms are paired.
"""
import argparse
import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))
sys.path.insert(0, _DIR)

import numpy as np
import tensorflow as tf
import keras

from gpu_config import setup_gpu
from kepin_losses import make_kepin_loss
from kepin_training import rmse_np, mae_np, physics_metrics_np, SEED
from kepin_cmapss_optimized import (EnhancedKePINTrainer, augment_time_series,
                                    smooth_rul_labels, CMAPSS_CONFIGS)
from dk_data import load_cmapss, engine_split
from dk_domain import DOMAINS, TRAIN_CFG, load_domain
from dk_model import DynamicKePIN


def cmapss_score(y_true, y_pred):
    d = np.ravel(y_pred) - np.ravel(y_true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def predict_batched(model, inputs, batch=256):
    """RUL predictions and per-sample Koopman outputs, in chunks to bound GPU memory."""
    n = len(inputs[0]) if isinstance(inputs, tuple) else len(inputs)
    preds, outs = [], {}
    for i in range(0, n, batch):
        chunk = tuple(v[i:i + batch] for v in inputs) if isinstance(inputs, tuple) else inputs[i:i + batch]
        p, ko = model(chunk, training=False)
        preds.append(p.numpy())
        for k in ("health", "health_trajectory", "wear_rate", "adapt_log_rho", "adapt_theta", "regime_probs",
                  "forecast_health_index", "hyper_mode_gate"):
            if k in ko:
                outs.setdefault(k, []).append(ko[k].numpy())
    return np.concatenate(preds), {k: np.concatenate(v) for k, v in outs.items()}


def metrics(y_true, y_pred):
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-10
    return {"rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "r2": float(1 - ss_res / ss_tot),
            "score": cmapss_score(y_true, y_pred)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"] + DOMAINS)
    p.add_argument("--koopman", default="eig", choices=["svd", "eig", "none"])
    p.add_argument("--mode", default="static", choices=["static", "exo", "health", "both", "hyper"])
    p.add_argument("--hyper_age", action="store_true",
                   help="hyper mode: the hypernetwork also reads the engine's age (flights flown so far)")
    p.add_argument("--hist_health", action="store_true",
                   help="original KePIN: a health index h from 42 engine-history features (past flights only, "
                        "per-condition z-scored) is the ONLY context of the hypernetwork Koopman (+ age with "
                        "--hyper_age); train it label-free with --regime_loss mono")
    p.add_argument("--head_ctx", action="store_true",
                   help="control for --hist_health: static K, and [h, age] go directly into the KePIN head")
    p.add_argument("--hyper_rank", action="store_true",
                   help="hyper mode: per-mode ReLU gates can switch Koopman modes off (adaptive rank)")
    p.add_argument("--protocol", default="clean", choices=["paper", "clean"])
    p.add_argument("--run_id", type=int, default=0)
    p.add_argument("--head", default="kepin", choices=["kepin", "ttf", "wear"],
                   help="kepin: original RUL head; ttf: RUL = Koopman steps until projected health crosses 0")
    p.add_argument("--future_loss", type=float, default=0.0,
                   help="weight of the long-horizon Koopman loss on real same-engine future windows (0 = off)")
    p.add_argument("--future_hmax", type=int, default=60, help="largest horizon (flights) for --future_loss")
    p.add_argument("--future_subspace", type=int, default=0,
                   help="k > 0: apply --future_loss only to 2k dedicated latent dims with a growth term")
    p.add_argument("--cond_norm", action="store_true",
                   help="learned per-operating-condition input scale/offset before (in-model) EMA smoothing")
    p.add_argument("--koop_control", action="store_true",
                   help="Koopman with control: z = s + C*condition; Koopman losses and RUL head use s")
    p.add_argument("--no_koopman_loss", action="store_true",
                   help="ablation: keep the architecture, drop the Koopman one-step, "
                        "multi-step and spectral losses")
    p.add_argument("--health_map", default="free", choices=["free", "ordered"],
                   help="ordered: lower health can only speed decay up (healthy slow, failing fast)")
    p.add_argument("--head_state", type=int, default=0,
                   help="H > 0: RUL head also reads the current state carried H flights ahead by K(h)")
    p.add_argument("--regime_loss", default="none", choices=["none", "mono", "sup"],
                   help="tie health h to degradation: monotone over life, or supervised by RUL")
    p.add_argument("--regime_weight", type=float, default=1.0)
    p.add_argument("--health_source", default="latent", choices=["latent", "signal_a"],
                   help="h from the latent state (learned MLP) or from condition-normalised Signal A")
    p.add_argument("--adapt", default="gate", choices=["gate", "mixture", "engine"],
                   help="how the operator adapts per engine: health-gated decay, mixture of 3 regime "
                        "operators, or per-engine eigenvalues fitted on the window")
    p.add_argument("--dynamics", default="linear", choices=["linear", "nl_static", "nl_dynamic"],
                   help="wear model latent dynamics: linear K; K + bounded MLP; K(h) with h re-read "
                        "from the state at every forecast step")
    p.add_argument("--encoder", default="deep", choices=["deep", "linear"],
                   help="deep KePIN encoder, or a plain linear map of the condition-normalised sensors "
                        "(keeps the degradation curve; wear model only)")
    p.add_argument("--frozen_encoder", action="store_true",
                   help="load the self-supervised encoder runs/ssl/<dataset>/run<id>_encoder.npz "
                        "(pretrain_ssl.py, no RUL labels) and freeze it; only K and the wear head train")
    p.add_argument("--forecast_head", action="store_true",
                   help="head reads the state carried 1, 5, 10, 25, 50 flights ahead by the engine's operator")
    p.add_argument("--gap_init", type=float, default=-4.0,
                   help="initial decay-gap logit c (gap = 2*sigmoid(c)); -4 -> 0.036, 0 -> 1.0")
    p.add_argument("--regime_margin", type=float, default=0.5,
                   help="mono: h must drop by at least margin x (share of life passed)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny subset, for a quick wiring test")
    p.add_argument("--out", default=os.path.join(_DIR, "runs"))
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    if args.koopman == "svd" and args.mode != "static":
        p.error("the svd layer is only used as the static reference")

    setup_gpu(mixed_precision=False, xla=False, verbose=True)
    domain = args.dataset in DOMAINS   # cross-domain forecasting (original KePIN's multi-domain benchmark)
    if domain:
        if args.cond_norm or args.koop_control or args.regime_loss != "none" or args.head != "kepin":
            p.error("cross-domain datasets: original KePIN pipeline only (no cond_norm / regime loss / wear head)")
        tc = TRAIN_CFG[args.dataset]
        cfg = dict(epochs=tc["epochs"], patience=tc["patience"], lr=tc["lr"], batch_size=tc["batch_size"],
                   min_lr=1e-6, warmup_epochs=5, swa_start_frac=0.75, mixup_alpha=0.0, noise_std=0.0,
                   curriculum_warmup=15, label_smooth_sigma=0.0, stride=tc["stride"])
    else:
        cfg = dict(CMAPSS_CONFIGS[f"CMAPSS_{args.dataset}"])
    if args.epochs:
        cfg["epochs"] = args.epochs
    arm = f"{args.koopman}_{args.mode}" + ("" if args.head == "kepin" else f"_{args.head}") + \
        ("_nokooploss" if args.no_koopman_loss else "") + (f"_future{args.future_loss:g}" if args.future_loss > 0 else "") + \
        (f"_sub{args.future_subspace}" if args.future_loss > 0 and args.future_subspace else "") + \
        ("_cn" if args.cond_norm else "") + ("_ctrl" if args.koop_control else "") + \
        ("_ord" if args.health_map == "ordered" else "") + \
        ("" if args.adapt == "gate" else {"mixture": "_mix", "engine": "_eng"}[args.adapt]) + \
        ("_sigA" if args.health_source == "signal_a" else "") + \
        {"linear": "", "nl_static": "_nls", "nl_dynamic": "_nld"}[args.dynamics] + \
        ("_frz" if args.frozen_encoder else "") + ("_lin" if args.encoder == "linear" else "") + \
        ("_hh" if args.hist_health else "") + ("_hctx" if args.head_ctx else "") + \
        ("_age" if args.hyper_age else "") + ("_rank" if args.hyper_rank else "") + \
        (f"_gap{args.gap_init:g}" if args.gap_init != -4.0 else "") + \
        (f"_hs{args.head_state}" if args.head_state else "") + ("_fc" if args.forecast_head else "") + \
        ("" if args.regime_loss == "none" else f"_{args.regime_loss}" +
         ("" if args.regime_weight == 1.0 else f"{args.regime_weight:g}"))
    cond_aware = args.cond_norm or args.koop_control
    regime = args.koopman == "eig" and args.mode == "health" and args.cond_norm
    cn_wear = (args.head == "wear" and args.koopman == "eig" and args.mode in ("static", "health")
               and args.cond_norm and not args.koop_control)
    if cond_aware and not cn_wear and (not ((args.koopman == "svd" and args.mode == "static") or
                                            regime) or args.head != "kepin"):
        p.error("--cond_norm/--koop_control: original KePIN (--koopman svd --mode static), or the "
                "regime-adaptive model (--koopman eig --mode health --cond_norm)")
    if args.health_source != "latent" and not (regime or cn_wear):
        p.error("--health_source signal_a: regime-adaptive model or --head wear with --cond_norm")
    if (args.head_state or args.regime_loss != "none" or args.health_map != "free"
            or args.gap_init != -4.0 or args.adapt != "gate"
            or args.forecast_head) and not regime and not (args.hist_health and args.regime_loss == "mono"
                                                          and args.head_state == 0 and args.health_map == "free"):
        p.error("--health_map/--head_state/--regime_loss need --koopman eig --mode health --cond_norm")
    if args.adapt != "gate" and (args.health_source != "latent" or args.health_map != "free"
                                 or args.gap_init != -4.0):
        p.error("--adapt mixture/engine use the learned health head and no decay gap")
    if args.adapt == "engine" and args.regime_loss != "none":
        p.error("--adapt engine has no health index; use --regime_loss none")
    if args.future_loss > 0 and (args.koopman == "none" or args.mode != "static" or args.head != "kepin"):
        p.error("--future_loss is implemented for the static Koopman operator with the KePIN head")
    excluded = ("koopman", "multi_step", "spectral") if (args.no_koopman_loss or args.koopman == "none") else ()
    run_dir = os.path.join(args.out, args.protocol, args.dataset, arm)
    for attempt in range(20):   # parallel workers race on the shared (NFS) disk: retry
        try:
            os.makedirs(run_dir, exist_ok=True)
        except OSError:
            pass
        if os.path.isdir(run_dir):
            break
        time.sleep(0.5 + np.random.rand())
    else:
        raise OSError(f"could not create {run_dir}")
    tag = f"run{args.run_id}"
    print(f"=== {args.dataset} | arm={arm} | protocol={args.protocol} | run={args.run_id} ===", flush=True)

    D = load_domain(args.dataset) if domain else load_cmapss(args.dataset)
    X_tr, Y_tr, X_te, Y_te = D["X_train"], D["Y_train"], D["X_test"], D["Y_test"]
    if domain:   # train on a [0, 1] target (as KePIN's multi-domain runs); metrics in original units
        y_min, y_rng = float(Y_tr.min()), max(float(Y_tr.max() - Y_tr.min()), 1e-6)
        Y_te_orig = Y_te.copy()
        Y_tr, Y_te = (Y_tr - y_min) / y_rng, (Y_te - y_min) / y_rng
    nc = D["n_conditions"]           # one-hot columns actually used (1 on FD001/FD003)
    U_tr = D["U_train"][..., :nc].reshape(len(X_tr), -1)
    U_te = D["U_test"][..., :nc].reshape(len(X_te), -1)
    if cond_aware:   # raw (unsmoothed) windows: the model applies the condition step, then EMA
        X_tr, X_te = D["X_train_raw"], D["X_test_raw"]
    uses_exo = args.mode in ("exo", "both") or cond_aware
    exo_dim = D["n_conditions"] if uses_exo else 0
    if (args.hyper_age or args.hyper_rank) and args.mode != "hyper":
        p.error("--hyper_age/--hyper_rank need --mode hyper")
    if args.mode == "hyper" and (cond_aware or args.koopman != "eig" or args.head != "kepin"):
        p.error("--mode hyper: original KePIN pipeline (eig Koopman, KePIN head, no cond_norm)")
    if args.head_ctx and not (args.hist_health and args.mode == "static" and args.koopman == "eig"):
        p.error("--head_ctx is the control arm: --hist_health --koopman eig --mode static")
    if args.hist_health:  # second input: 42 engine-history features + age/100
        if not ((args.mode == "hyper") or args.head_ctx):
            p.error("--hist_health needs --mode hyper (or the --head_ctx control)")
        U_tr = np.concatenate([D["hist_train"], D["age_train"][:, None] / 100.0], 1).astype(np.float32)
        U_te = np.concatenate([D["hist_test"], D["age_test"][:, None] / 100.0], 1).astype(np.float32)
        uses_exo = True
    elif args.hyper_age:   # the second input carries the engine's age (flights flown), scaled
        U_tr, U_te = D["age_train"][:, None] / 100.0, D["age_test"][:, None] / 100.0
        uses_exo = True
    if args.mode in ("exo", "both") and exo_dim < 2:
        p.error(f"{args.dataset} has a single operating condition; exo mode is meaningless")

    if args.protocol == "clean":
        tr_mask, va_mask = engine_split(D["units_train"], val_frac=0.2, seed=0)
    else:
        tr_mask, va_mask = np.ones(len(X_tr), bool), None
    if domain and cfg["stride"] > 1:   # heavily overlapping windows: fit on every k-th (validation untouched)
        tr_mask = tr_mask & (np.arange(len(X_tr)) % cfg["stride"] == 0)
    if args.smoke:
        idx = np.where(tr_mask)[0][:2000]
        tr_mask = np.isin(np.arange(len(X_tr)), idx)
        if va_mask is not None:
            va_mask = np.isin(np.arange(len(X_tr)), np.where(va_mask)[0][:500])

    def inputs(X, U):
        return (X, U) if uses_exo else X

    X_fit, Y_fit, U_fit = X_tr[tr_mask], Y_tr[tr_mask], U_tr[tr_mask]
    if va_mask is not None:
        val_in, Y_val = inputs(X_tr[va_mask], U_tr[va_mask]), Y_tr[va_mask]
    else:
        val_in, Y_val = inputs(X_te, U_te), Y_te
    test_in = inputs(X_te, U_te)
    print(f"  train windows={len(X_fit)}  val windows={len(Y_val)}  test engines={len(Y_te)}  "
          f"features={X_tr.shape[2]}  conditions={D['n_conditions']}", flush=True)

    np.random.seed(SEED)
    Y_fit_s = Y_fit if domain else smooth_rul_labels(Y_fit, sigma=cfg["label_smooth_sigma"])

    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]
    if domain:   # KePIN's own size rule, as in its multi-domain runs (build_kepin_model with arch_config=None)
        from kepin_model import auto_configure
        cfg["arch_override"] = auto_configure(n_feat, seq_len, len(X_fit))
    arch = cfg["arch_override"].copy()
    arch["kernels"] = [min(k, seq_len) for k in arch["kernels"]]
    arch["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch["kernels"]]

    tf.random.set_seed(SEED + args.run_id * 100)
    np.random.seed(SEED + args.run_id * 100)
    X_aug, Y_aug = augment_time_series(inputs(X_fit, U_fit), Y_fit_s,
                                       noise_std=cfg["noise_std"] * 0.5, seed=SEED + args.run_id)

    cond_stats = sig_a_stats = None
    if args.cond_norm:   # per-condition mean/std of every raw feature over all training time steps
        xs = X_fit.reshape(-1, n_feat)
        cs = D["U_train"][tr_mask][..., :exo_dim].reshape(-1, exo_dim).argmax(1)
        mu = np.stack([xs[cs == c].mean(0) for c in range(exo_dim)])
        sd = np.maximum(np.stack([xs[cs == c].std(0) for c in range(exo_dim)]), 0.1)  # floor: near-constant
        cond_stats = (mu.astype(np.float32), sd.astype(np.float32))                    # features stay small
        # Signal A uses the SAME floored std as cond_norm: with the exact std, near-constant sensors
        # (std 1e-4..1e-3) are swamped by the training noise (~0.009), up to 100x their spread
        sig_a_stats = cond_stats
    model = DynamicKePIN((seq_len, n_feat), arch_config=arch, n_train=len(X_fit),
                         n_active_losses=(4 if domain else 7) - len(excluded), koopman=args.koopman,
                         koopman_mode=args.mode, exo_dim=exo_dim, head=args.head,
                         cond_norm=args.cond_norm, koop_control=args.koop_control,
                         cond_stats=cond_stats, ema_alpha=D["ema_alpha"],
                         health_map=args.health_map,
                         head_state=[1, 5, 10, 25, 50] if args.forecast_head else args.head_state,
                         health_source=args.health_source, gap_init=args.gap_init, adapt=args.adapt,
                         dynamics=args.dynamics, encoder=args.encoder,
                         hyper_ctx_dim=1 if args.hyper_age else 0, hyper_rank=args.hyper_rank,
                         hist_health=args.hist_health, use_age=args.hyper_age, head_ctx=args.head_ctx,
                         sig_a_stats=sig_a_stats, hist_dim=D["hist_train"].shape[1])
    _ = model(tuple(tf.constant(v[:2]) for v in test_in) if uses_exo
              else tf.constant(test_in[:2]), training=False)
    if args.frozen_encoder:
        if not cn_wear:
            p.error("--frozen_encoder is for the condition-aware wear model")
        enc_path = os.path.join(_DIR, "runs", "ssl", args.dataset, f"run{args.run_id}_encoder.npz")
        for _ in range(360):                       # the pre-training may still be running: wait <= 3 h
            if os.path.exists(enc_path):
                break
            time.sleep(30)
        time.sleep(5)                              # let the writer finish
        model.load_frozen_encoder(enc_path)
        print(f"  frozen self-supervised encoder: {enc_path}", flush=True)
    if args.health_source == "signal_a":
        model.set_signal_a_scale((X_fit, U_fit))
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"  params={n_params:,}", flush=True)

    loss_fn = make_kepin_loss(loss_weights_layer=model.loss_weight_layer,
                              use_auto_weights=True, domain_mode="forecasting" if domain else "degradation",
                              exclude=excluded)
    optimizer = keras.optimizers.Adam(learning_rate=float(cfg["lr"]), clipnorm=1.0)
    trainer_kw = dict(clip_norm=2.0, warmup_epochs=cfg["warmup_epochs"],
                      curriculum_warmup=cfg["curriculum_warmup"], mixup_alpha=cfg["mixup_alpha"],
                      swa_start_frac=cfg["swa_start_frac"])
    if args.future_loss > 0:
        from future_trainer import FutureKoopmanTrainer, FuturePairSampler
        sampler = FuturePairSampler(X_aug, D["units_train"][tr_mask], h_max=args.future_hmax,
                                    seed=SEED + args.run_id)
        if args.future_subspace:
            # per-dim decay of the dedicated future subspace; init rho ~ 0.95
            model.future_rho_logit = model.add_weight(
                name="future_rho_logit", shape=(args.future_subspace,),
                initializer=keras.initializers.Constant(3.0), trainable=True)
        trainer = FutureKoopmanTrainer(model, loss_fn, optimizer, sampler=sampler,
                                       future_weight=args.future_loss, h_max=args.future_hmax,
                                       subspace_k=args.future_subspace, **trainer_kw)
    elif args.regime_loss != "none":
        from regime_trainer import RegimeKoopmanTrainer, RegimePairSampler
        rul_cap = float(np.max(Y_tr))                   # 125 on C-MAPSS
        sampler = RegimePairSampler(X_aug[0], X_aug[1], Y_fit, D["units_train"][tr_mask],
                                    rul_cap=rul_cap, seed=SEED + args.run_id)
        trainer = RegimeKoopmanTrainer(model, loss_fn, optimizer, sampler=sampler,
                                       regime_loss=args.regime_loss, regime_weight=args.regime_weight,
                                       margin=args.regime_margin, **trainer_kw)
    else:
        trainer = EnhancedKePINTrainer(model, loss_fn, optimizer, **trainer_kw)
    t0 = time.time()
    history, swa_w = trainer.fit_enhanced(
        X_aug, Y_aug, val_in, Y_val, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        patience=cfg["patience"], initial_lr=cfg["lr"], min_lr=cfg["min_lr"],
        noise_std=cfg["noise_std"], verbose=args.verbose)
    train_time = time.time() - t0

    # choose best-epoch vs SWA weights on the validation set (test set for "paper")
    best_w = model.get_weights()
    val_best = rmse_np(Y_val, predict_batched(model, val_in)[0])
    chosen = "best"
    if swa_w is not None:
        model.set_weights(swa_w)
        val_swa = rmse_np(Y_val, predict_batched(model, val_in)[0])
        if val_swa < val_best:
            chosen = "swa"
        else:
            model.set_weights(best_w)
    y_pred, kout = predict_batched(model, test_in)
    y_pred = y_pred.ravel()
    if domain:   # back to original units; correlations below use the normalised target (same values)
        y_pred = y_pred * y_rng + y_min

    res = {"dataset": args.dataset, "arm": arm, "protocol": args.protocol, "run_id": args.run_id,
           **(metrics(Y_te_orig, y_pred) if domain else metrics(Y_te, y_pred)),
           "val_rmse": float(min(val_best, val_swa) if swa_w is not None else val_best),
           **({"val_rmse_units": "normalised [0,1] target", "y_min": y_min, "y_range": y_rng,
               "train_stride": cfg["stride"], "arch": cfg["arch_override"]} if domain else {}),
           "weights_used": chosen, "epochs_trained": len(history["epoch"]),
           "train_time_s": train_time, "sec_per_epoch": train_time / max(1, len(history["epoch"])),
           "params": n_params, "gpu": os.environ.get("SLURM_JOB_PARTITION", "local"),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": args.smoke}
    if domain:
        res.pop("score", None)
    else:
        mono, slope = physics_metrics_np(Y_te, y_pred.reshape(-1, 1))
        res.update(mono_violation=float(mono), slope_rmse=float(slope))
    if regime and "health" in kout:   # regime check on the test engines: does h track wear?
        h_te = kout["health"].ravel()
        res["health_corr_rul"] = float(np.corrcoef(h_te, Y_te.ravel())[0, 1])
        res["health_std"] = float(h_te.std())
    if regime:
        # how much the operator differs between engines: per-engine 10-step keep factor rho^10
        keep10 = np.exp(10 * kout["adapt_log_rho"])                              # (engines, modes)
        near, healthy = Y_te.ravel() <= 20, Y_te.ravel() > 100
        if near.any() and healthy.any():
            res["keep10_nearfail_over_healthy"] = float(np.median(
                np.median(keep10[near], 0) / np.maximum(np.median(keep10[healthy], 0), 1e-12)))
        if args.adapt == "mixture":
            pi = kout["regime_probs"]
            res["regime_share"] = [float(v) for v in np.bincount(pi.argmax(1), minlength=3) / len(pi)]
            res["regime_mean_rul"] = [float(Y_te.ravel()[pi.argmax(1) == r].mean()) if (pi.argmax(1) == r).any()
                                      else None for r in range(3)]
            res["regime_decay_offset_mean"] = [float(v) for v in model.koopman.reg_ds.numpy().mean(1)]
        if args.adapt == "engine":
            res["engine_mix_alpha_mean"] = float(tf.sigmoid(model.koopman.engine_mix_logit).numpy().mean())
        if args.adapt == "gate" and args.health_map == "ordered":
            gap = model.koopman.health_gap().numpy()
            res["decay_gap_mean"] = float(gap.mean())
            res["decay_gap_max"] = float(gap.max())
        if args.health_source == "signal_a":   # the readable regime formula
            res["sigA_w"] = dict(zip(["mean", "var", "skew", "kurt"], map(float, model.sa_w.numpy())))
            res["sigA_b"] = float(model.sa_b.numpy())
        if getattr(trainer, "regime_log", None):
            last = np.array(trainer.regime_log[-200:])
            res["regime_loss_end"], res["regime_h_drop_end"] = float(last[:, 0].mean()), float(last[:, 1].mean())
    if cn_wear and args.health_source == "signal_a":   # the readable wear-rate formula
        from dk_model import SA_LR
        res["sigA_wear_w"] = dict(zip(["mean", "var", "skew", "kurt"], map(float, SA_LR * model.sa_w.numpy())))
        res["sigA_wear_b"] = float(SA_LR * model.sa_b.numpy())
    if args.hist_health and "health" in kout:   # is h a health index, and does K follow it?
        h_te = kout["health"].ravel()
        res["hist_h_corr_rul"] = float(np.corrcoef(h_te, Y_te.ravel())[0, 1])
        res["hist_h_std"] = float(h_te.std())
        if args.mode == "hyper":
            rho_m = np.exp(kout["adapt_log_rho"]).mean(1)
            res["hyper_mean_rho_corr_h"] = float(np.corrcoef(rho_m, h_te)[0, 1])
            if "hyper_mode_gate" in kout:
                act = (kout["hyper_mode_gate"] > 1e-3).sum(1)
                res["hyper_active_modes_corr_h"] = float(np.corrcoef(act, h_te)[0, 1]) if act.std() > 0 else 0.0
        if getattr(trainer, "regime_log", None):
            last = np.array(trainer.regime_log[-200:])
            res["regime_loss_end"], res["regime_h_drop_end"] = float(last[:, 0].mean()), float(last[:, 1].mean())
    if args.mode == "hyper":   # how much the operator differs between engines, and why
        rho = np.exp(kout["adapt_log_rho"])                                      # (engines, modes)
        res["hyper_rho_spread"] = float(rho.std(0).mean())
        res["hyper_mean_rho_corr_rul"] = float(np.corrcoef(rho.mean(1), Y_te.ravel())[0, 1])
        res["hyper_mean_rho_corr_age"] = float(np.corrcoef(rho.mean(1), D["age_test"])[0, 1])
        if "hyper_mode_gate" in kout:
            active = (kout["hyper_mode_gate"] > 1e-3).sum(1)
            res["hyper_active_modes_mean"] = float(active.mean())
            res["hyper_active_modes_corr_rul"] = float(np.corrcoef(active, Y_te.ravel())[0, 1]) if active.std() > 0 else 0.0
    if "forecast_health_index" in kout:   # does h fall along the forecast? (start vs 50 flights ahead)
        fh = kout["forecast_health_index"]
        res["forecast_h_start_median"] = float(np.median(fh[:, 0]))
        res["forecast_h_at50_median"] = float(np.median(fh[:, 49]))
        res["forecast_h_corr_rul"] = float(np.corrcoef(fh[:, 0], Y_te.ravel())[0, 1])
    if "wear_rate" in kout:
        res["wear_rate_corr_rul"] = float(np.corrcoef(kout["wear_rate"].ravel(), Y_te.ravel())[0, 1])
    print(json.dumps(res, indent=2), flush=True)

    if not args.smoke:
        model.save_weights(os.path.join(run_dir, f"{tag}.weights.h5"))
    extra = {"health": kout["health"].ravel()} if "health" in kout else {}
    if "health_trajectory" in kout:     # projected health per test engine, steps 0..125
        extra["health_trajectory"] = kout["health_trajectory"]
    if "wear_rate" in kout:
        extra["wear_rate"] = kout["wear_rate"].ravel()
    if regime:
        extra["adapt_log_rho"], extra["adapt_theta"] = kout["adapt_log_rho"], kout["adapt_theta"]
    if "regime_probs" in kout:
        extra["regime_probs"] = kout["regime_probs"]
    if "forecast_health_index" in kout:
        extra["forecast_health_index"] = kout["forecast_health_index"]
    if regime and args.adapt == "gate" and args.health_map == "ordered":
        extra["decay_gap"] = model.koopman.health_gap().numpy()
        extra["decay_logit_healthy"] = model.koopman.s.numpy()
    if getattr(trainer, "regime_log", None):
        extra["regime_log"] = np.array(trainer.regime_log, np.float32)
    np.savez(os.path.join(run_dir, f"{tag}_predictions.npz"), y_true=(Y_te_orig if domain else Y_te).ravel(),
             y_pred=y_pred, **extra)
    keep = {k: v for k, v in history.items()
            if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])}
    import pandas as pd
    pd.DataFrame(keep).to_csv(os.path.join(run_dir, f"{tag}_history.csv"), index=False)
    with open(os.path.join(run_dir, f"{tag}_result.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
