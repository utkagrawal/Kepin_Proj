# -*- coding: utf-8 -*-
"""Long-horizon "future" Koopman loss on top of the original KePIN trainer.

KePIN's own Koopman losses only check 1-3 steps ahead inside one 30-flight window.
Here, every training step also draws a small batch of (window at flight t,
window of the SAME engine at flight t+h), h in [5, h_max], and asks the Koopman
operator to carry the current latent state h steps ahead onto the latent state
the encoder produces for the real future window:

    L_future = mean || K^h z_T(t) - stopgrad(z_T(t+h)) ||^2

This teaches the shared encoder + operator the real long-term degradation path of
each engine, which a single window does not contain.

Subspace variant (subspace_k = k > 0). The full-latent version squeezed the shared latent
into ~1 slow mode (a stable operator cannot represent growing wear) and hurt accuracy.
Here the loss only touches 2k dedicated latent dims: x = z[:k] (state), r = z[k:2k] (rate),
with its own small diagonal Koopman rule that includes a growth term (an eigenvalue-1
integrator driven by r):
    x(t+h) = rho^h * x(t) + r(t) * (1 - rho^h) / (1 - rho)        (-> x + r*h as rho -> 1)
    with r(t) = 0.02 * z[k:2k] (keeps the initial drift small; the encoder can scale it up)
    L_future = mean || x_pred - stopgrad(x(t+h)) ||^2
rho (k values in (0,1)) is a trainable model weight `future_rho_logit`. The rest of the
latent, KePIN's own Koopman operator and the RUL head are left alone. Everything else (mixup, SWA,
warm-up, all original losses) is unchanged: the parent train steps are reproduced
and only `future_weight * L_future` is added to the total.
"""
import numpy as np
import tensorflow as tf

from kepin_cmapss_optimized import EnhancedKePINTrainer


class FuturePairSampler:
    """Random (window, same-engine future window, h) batches from the training set."""

    def __init__(self, X, units, batch=64, h_min=5, h_max=60, seed=0):
        self.X = X
        self.batch = batch
        self.h_min, self.h_max = h_min, h_max
        self.rng = np.random.default_rng(seed)
        # windows of one engine are contiguous and in time order (GDS sliding windows)
        remaining = np.zeros(len(units), dtype=int)
        for u in np.unique(units):
            idx = np.where(units == u)[0]
            remaining[idx] = np.arange(len(idx))[::-1]
        self.remaining = remaining
        self.pool = np.where(remaining >= 1)[0]

    def next(self):
        i = self.rng.choice(self.pool, self.batch)
        h = self.rng.integers(self.h_min, self.h_max + 1, self.batch)
        h = np.minimum(h, self.remaining[i])           # stay inside the same engine's life
        return (tf.constant(self.X[i]), tf.constant(self.X[i + h]), tf.constant(h, tf.int32))


class FutureKoopmanTrainer(EnhancedKePINTrainer):

    def __init__(self, *args, sampler=None, future_weight=1.0, h_max=60, subspace_k=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = sampler
        self.future_weight = float(future_weight)
        self.h_max = int(h_max)
        self.subspace_k = int(subspace_k)

    def _K(self):
        kp = self.model.koopman
        return kp.koopman_matrix() if hasattr(kp, "koopman_matrix") else kp._get_K()

    def future_loss(self, Xa, Xb, h):
        if self.subspace_k:
            return self._future_loss_subspace(Xa, Xb, h)
        K = tf.cast(self._K(), tf.float32)
        powers = [K]
        for _ in range(self.h_max - 1):                  # K^1 .. K^h_max (d x d each, cheap)
            powers.append(tf.matmul(powers[-1], K))
        P = tf.gather(tf.stack(powers), h - 1)            # (B, d, d)
        za = tf.cast(self.model.encode(Xa, training=True)[:, -1], tf.float32)
        zb = tf.stop_gradient(tf.cast(self.model.encode(Xb, training=True)[:, -1], tf.float32))
        pred = tf.einsum("bij,bj->bi", P, za)             # column convention z_{t+1} = K z_t
        return tf.reduce_mean(tf.square(pred - zb))

    def _future_loss_subspace(self, Xa, Xb, h):
        k = self.subspace_k
        za = tf.cast(self.model.encode(Xa, training=True)[:, -1], tf.float32)
        zb = tf.stop_gradient(tf.cast(self.model.encode(Xb, training=True)[:, -1, :k], tf.float32))
        # rate scaled by 0.02: latent dims are ~unit variance (BatchNorm), so an unscaled rate
        # would add up to ~h units of drift at initialisation and swamp training
        x, r = za[:, :k], 0.02 * za[:, k:2 * k]
        rho = tf.sigmoid(self.model.future_rho_logit)                      # (k,)
        hf = tf.cast(h, tf.float32)[:, None]
        rho_h = tf.exp(hf * tf.math.log(rho))
        one_minus = 1.0 - rho
        growth = tf.where(one_minus < 1e-4, hf, (1.0 - rho_h) / tf.maximum(one_minus, 1e-4))
        pred = rho_h * x + r * growth
        return tf.reduce_mean(tf.square(pred - zb))

    def _apply(self, tape, total):
        gradients = tape.gradient(total, self.model.trainable_variables)
        global_norm = tf.linalg.global_norm(gradients)
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        s_factor = tf.minimum(1.0, self.clip_norm / (global_norm + 1e-8))
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return global_norm, s_factor

    @staticmethod
    def _f32(koopman_out):
        return {k: tf.cast(v, tf.float32) if v.dtype != tf.complex64 else v
                for k, v in koopman_out.items() if isinstance(v, tf.Tensor)}

    @tf.function(jit_compile=False)
    def _step(self, X_batch, Y_batch, Xa, Xb, h):
        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(X_batch, training=True)
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            total_loss, loss_dict = self.loss_fn(tf.cast(Y_batch, tf.float32), rul_pred_f32,
                                                 self._f32(koopman_out))
            lf = self.future_loss(Xa, Xb, h)
            total_loss = total_loss + self.future_weight * lf
        loss_dict["future"] = lf
        g, s = self._apply(tape, total_loss)
        return total_loss, loss_dict, rul_pred_f32, g, s

    @tf.function(jit_compile=False)
    def _step_mixup(self, X1, Y1, X2, Y2, lam, Xa, Xb, h):
        lam_y = tf.reshape(lam[:, 0, 0], (-1, 1))
        Y_mix = lam_y * Y1 + (1.0 - lam_y) * Y2
        if isinstance(X1, (tuple, list)):
            inputs_mix = (lam * X1[0] + (1.0 - lam) * X2[0], lam_y * X1[1] + (1.0 - lam_y) * X2[1])
        else:
            inputs_mix = lam * X1 + (1.0 - lam) * X2
        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(inputs_mix, training=True)
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            total_loss, loss_dict = self.loss_fn(tf.cast(Y_mix, tf.float32), rul_pred_f32,
                                                 self._f32(koopman_out))
            lf = self.future_loss(Xa, Xb, h)             # future pairs are never mixed
            total_loss = total_loss + self.future_weight * lf
        loss_dict["future"] = lf
        g, s = self._apply(tape, total_loss)
        return total_loss, loss_dict, rul_pred_f32, g, s

    # the parent fit loop calls these two; a fresh future batch is drawn every step
    def train_step(self, X_batch, Y_batch):
        return self._step(X_batch, Y_batch, *self.sampler.next())

    def train_step_mixup(self, X1, Y1, X2, Y2, lam):
        return self._step_mixup(X1, Y1, X2, Y2, lam, *self.sampler.next())
