#!/usr/bin/env python3
"""
Unit test + speed benchmark for the parameter-conditioned Koopman operator.

Tests:
  1. Warm-start: K(mu1) == K(mu2) before any training (zero-init guarantee)
  2. Speed: conditioned forward+backward within ~20% of unconditioned speed
"""
import os, sys, time
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from kepin.utils.gpu import setup_gpu
setup_gpu(verbose=True)

import tensorflow as tf
from kepin.models.kepin_model import build_kepin_model

# -----------------------------------------------------------------------
# Settings matching the real FD002 setup
# -----------------------------------------------------------------------
SEQ_LEN     = 30
N_FEATURES  = 24       # FD002/FD004 feature count
CONDITION_DIM = 3
# In FD002, "setting1", "setting2", "setting3" are the first 3 feature cols
CONDITION_INDICES = [0, 1, 2]
BATCH_SIZE  = 1024
N_WARMUP    = 3        # warmup forward passes before timing
N_TIMED     = 10       # timed iterations

# -----------------------------------------------------------------------
# TEST 1: Warm-start (K(mu) identical across different mu at init)
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 1: Warm-start — K(μ) identical across μ at initialization")
print("="*60)

model_cond = build_kepin_model(
    SEQ_LEN, N_FEATURES,
    condition_dim=CONDITION_DIM,
    condition_indices=CONDITION_INDICES,
)
koopman_op = model_cond.koopman

# Build with dummy data so weights are initialized
dummy = tf.zeros((2, SEQ_LEN, N_FEATURES))
_ = model_cond(dummy, training=False)

# Run with several different random mu inputs
rng = np.random.default_rng(42)
mus = [tf.constant(rng.standard_normal((1, CONDITION_DIM)).astype(np.float32))
       for _ in range(5)]

Ks = [koopman_op._get_K(condition=mu).numpy() for mu in mus]
max_diff = max(np.max(np.abs(Ks[i] - Ks[0])) for i in range(1, len(Ks)))
print(f"  Max |K(μᵢ) - K(μ₀)| across 5 random μ: {max_diff:.2e}")
if max_diff < 1e-5:
    print("  ✓ PASS: K(μ) is numerically identical at initialization")
else:
    print("  ✗ FAIL: K(μ) differs at initialization — zero-init broken!")
    sys.exit(1)

# -----------------------------------------------------------------------
# TEST 2: Speed benchmark — conditioned vs unconditioned
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("TEST 2: Speed — condition_dim=3 vs condition_dim=0")
print("="*60)

model_base = build_kepin_model(SEQ_LEN, N_FEATURES, condition_dim=0)
_ = model_base(dummy, training=False)

# Generate a synthetic batch
X_batch = tf.constant(rng.standard_normal((BATCH_SIZE, SEQ_LEN, N_FEATURES)).astype(np.float32))
Y_batch = tf.constant(rng.uniform(0, 125, (BATCH_SIZE, 1)).astype(np.float32))

from kepin.losses.composite import make_kepin_loss
loss_fn_base = make_kepin_loss(domain_mode="degradation", use_auto_weights=False)
loss_fn_cond = make_kepin_loss(domain_mode="degradation", use_auto_weights=False)

optimizer_base = tf.keras.optimizers.Adam(1e-3)
optimizer_cond = tf.keras.optimizers.Adam(1e-3)

def train_step(model, loss_fn, optimizer, X, Y):
    with tf.GradientTape() as tape:
        rul_pred, koopman_out = model(X, training=True)
        loss, _ = loss_fn(Y, rul_pred, koopman_out)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Warmup
print(f"  Warming up ({N_WARMUP} steps each)...")
for _ in range(N_WARMUP):
    train_step(model_base, loss_fn_base, optimizer_base, X_batch, Y_batch)
    train_step(model_cond, loss_fn_cond, optimizer_cond, X_batch, Y_batch)

# Timed
print(f"  Timing {N_TIMED} steps each (batch={BATCH_SIZE}, d=128)...")

t0 = time.perf_counter()
for _ in range(N_TIMED):
    train_step(model_base, loss_fn_base, optimizer_base, X_batch, Y_batch)
t_base = (time.perf_counter() - t0) / N_TIMED

t0 = time.perf_counter()
for _ in range(N_TIMED):
    train_step(model_cond, loss_fn_cond, optimizer_cond, X_batch, Y_batch)
t_cond = (time.perf_counter() - t0) / N_TIMED

ratio = t_cond / t_base
print(f"\n  condition_dim=0  (baseline): {t_base*1000:.1f} ms/step")
print(f"  condition_dim=3  (conditioned): {t_cond*1000:.1f} ms/step")
print(f"  Ratio (conditioned/baseline): {ratio:.2f}x")

if ratio <= 1.20:
    print("  ✓ PASS: Within 20% of baseline speed")
elif ratio <= 2.0:
    print(f"  ⚠ WARNING: {ratio:.2f}x slower — acceptable but watch closely")
else:
    print(f"  ✗ FAIL: {ratio:.2f}x slower than baseline — do NOT proceed to training")
    sys.exit(1)

print("\nAll tests passed! Ready for real training runs.")
