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
        for k in ("health", "health_trajectory", "wear_rate"):
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
    p.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    p.add_argument("--koopman", default="eig", choices=["svd", "eig", "none"])
    p.add_argument("--mode", default="static", choices=["static", "exo", "health", "both"])
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
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="tiny subset, for a quick wiring test")
    p.add_argument("--out", default=os.path.join(_DIR, "runs"))
    p.add_argument("--verbose", type=int, default=1)
    args = p.parse_args()
    if args.koopman == "svd" and args.mode != "static":
        p.error("the svd layer is only used as the static reference")

    setup_gpu(mixed_precision=False, xla=False, verbose=True)
    cfg = dict(CMAPSS_CONFIGS[f"CMAPSS_{args.dataset}"])
    if args.epochs:
        cfg["epochs"] = args.epochs
    arm = f"{args.koopman}_{args.mode}" + ("" if args.head == "kepin" else f"_{args.head}") + \
        ("_nokooploss" if args.no_koopman_loss else "") + (f"_future{args.future_loss:g}" if args.future_loss > 0 else "") + \
        (f"_sub{args.future_subspace}" if args.future_loss > 0 and args.future_subspace else "") + \
        ("_cn" if args.cond_norm else "") + ("_ctrl" if args.koop_control else "")
    cond_aware = args.cond_norm or args.koop_control
    if cond_aware and (args.koopman != "svd" or args.mode != "static" or args.head != "kepin"):
        p.error("--cond_norm/--koop_control are built on the original KePIN (--koopman svd --mode static)")
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

    D = load_cmapss(args.dataset)
    X_tr, Y_tr, X_te, Y_te = D["X_train"], D["Y_train"], D["X_test"], D["Y_test"]
    nc = D["n_conditions"]           # one-hot columns actually used (1 on FD001/FD003)
    U_tr = D["U_train"][..., :nc].reshape(len(X_tr), -1)
    U_te = D["U_test"][..., :nc].reshape(len(X_te), -1)
    if cond_aware:   # raw (unsmoothed) windows: the model applies the condition step, then EMA
        X_tr, X_te = D["X_train_raw"], D["X_test_raw"]
    uses_exo = args.mode in ("exo", "both") or cond_aware
    exo_dim = D["n_conditions"] if uses_exo else 0
    if args.mode in ("exo", "both") and exo_dim < 2:
        p.error(f"{args.dataset} has a single operating condition; exo mode is meaningless")

    if args.protocol == "clean":
        tr_mask, va_mask = engine_split(D["units_train"], val_frac=0.2, seed=0)
    else:
        tr_mask, va_mask = np.ones(len(X_tr), bool), None
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
    Y_fit_s = smooth_rul_labels(Y_fit, sigma=cfg["label_smooth_sigma"])

    arch = cfg["arch_override"].copy()
    seq_len, n_feat = X_tr.shape[1], X_tr.shape[2]
    arch["kernels"] = [min(k, seq_len) for k in arch["kernels"]]
    arch["kernels"] = [k if k % 2 == 1 else k - 1 for k in arch["kernels"]]

    tf.random.set_seed(SEED + args.run_id * 100)
    np.random.seed(SEED + args.run_id * 100)
    X_aug, Y_aug = augment_time_series(inputs(X_fit, U_fit), Y_fit_s,
                                       noise_std=cfg["noise_std"] * 0.5, seed=SEED + args.run_id)

    cond_stats = None
    if args.cond_norm:   # per-condition mean/std of every raw feature over all training time steps
        xs = X_fit.reshape(-1, n_feat)
        cs = D["U_train"][tr_mask][..., :exo_dim].reshape(-1, exo_dim).argmax(1)
        mu = np.stack([xs[cs == c].mean(0) for c in range(exo_dim)])
        sd = np.maximum(np.stack([xs[cs == c].std(0) for c in range(exo_dim)]), 0.1)  # floor: near-constant
        cond_stats = (mu.astype(np.float32), sd.astype(np.float32))                    # features stay small
    model = DynamicKePIN((seq_len, n_feat), arch_config=arch, n_train=len(X_fit),
                         n_active_losses=7 - len(excluded), koopman=args.koopman,
                         koopman_mode=args.mode, exo_dim=exo_dim, head=args.head,
                         cond_norm=args.cond_norm, koop_control=args.koop_control,
                         cond_stats=cond_stats, ema_alpha=D["ema_alpha"])
    _ = model(tuple(tf.constant(v[:2]) for v in test_in) if uses_exo
              else tf.constant(test_in[:2]), training=False)
    n_params = int(sum(np.prod(v.shape) for v in model.trainable_variables))
    print(f"  params={n_params:,}", flush=True)

    loss_fn = make_kepin_loss(loss_weights_layer=model.loss_weight_layer,
                              use_auto_weights=True, domain_mode="degradation",
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

    res = {"dataset": args.dataset, "arm": arm, "protocol": args.protocol, "run_id": args.run_id,
           **metrics(Y_te, y_pred),
           "val_rmse": float(min(val_best, val_swa) if swa_w is not None else val_best),
           "weights_used": chosen, "epochs_trained": len(history["epoch"]),
           "train_time_s": train_time, "sec_per_epoch": train_time / max(1, len(history["epoch"])),
           "params": n_params, "gpu": os.environ.get("SLURM_JOB_PARTITION", "local"),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "smoke": args.smoke}
    mono, slope = physics_metrics_np(Y_te, y_pred.reshape(-1, 1))
    res.update(mono_violation=float(mono), slope_rmse=float(slope))
    print(json.dumps(res, indent=2), flush=True)

    if not args.smoke:
        model.save_weights(os.path.join(run_dir, f"{tag}.weights.h5"))
    extra = {"health": kout["health"].ravel()} if "health" in kout else {}
    if "health_trajectory" in kout:     # projected health per test engine, steps 0..125
        extra["health_trajectory"] = kout["health_trajectory"]
    if "wear_rate" in kout:
        extra["wear_rate"] = kout["wear_rate"].ravel()
    np.savez(os.path.join(run_dir, f"{tag}_predictions.npz"), y_true=Y_te.ravel(), y_pred=y_pred, **extra)
    keep = {k: v for k, v in history.items()
            if k not in ("eigenvalues", "loss_weights") and len(v) == len(history["epoch"])}
    import pandas as pd
    pd.DataFrame(keep).to_csv(os.path.join(run_dir, f"{tag}_history.csv"), index=False)
    with open(os.path.join(run_dir, f"{tag}_result.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
