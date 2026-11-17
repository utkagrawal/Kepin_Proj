# -*- coding: utf-8 -*-
"""Regime loss for the regime-adaptive Koopman model (health index h drives the decay).

Without a training signal, h collapses to a constant (eig_health: spread across engines
~4e-7) or picks up the operating condition (Signal A: R^2 0.97 from the condition mix).
This trainer ties h to degradation. Every training step draws a small batch of pairs
(window of an engine at flight t, window of the SAME engine at flight t + dt), dt = 10..60,
and adds  regime_weight * L_regime  to KePIN's own losses:

  mono : h must fall over life. The later window must be lower by at least a margin
         proportional to the share of the engine's life that passed:
             L = mean relu( h(t+dt) - h(t) + margin * dt / life )
         The margin stops the trivial answer "h constant". Uses only time order and life
         length of training engines, never the RUL labels.
  sup  : reference arm: h is regressed on the (capped) RUL label, h ~ RUL / cap.
  none : pairs are not used (unconstrained h, as in Signal A / eig_health).

Both pair windows go through the encoder in ONE extra call (2 x batch windows), so the
overhead is one small forward/backward pass per step. Pairs are never mixed by mixup.
"""
import numpy as np
import tensorflow as tf

from kepin_cmapss_optimized import EnhancedKePINTrainer


class RegimePairSampler:
    """Random (window at t, same-engine window at t+dt) pairs from the training windows."""

    def __init__(self, X, U, Y, units, rul_cap, batch=32, dt_min=10, dt_max=60, seed=0):
        self.X, self.U = X, U
        self.y = (np.ravel(Y) / float(rul_cap)).astype(np.float32)
        self.batch, self.dt_min, self.dt_max = batch, dt_min, dt_max
        self.rng = np.random.default_rng(seed)
        # windows of one engine are contiguous and in time order (GDS sliding windows)
        remaining = np.zeros(len(units), dtype=int)
        life = np.zeros(len(units), dtype=np.float32)
        for u in np.unique(units):
            idx = np.where(units == u)[0]
            remaining[idx] = np.arange(len(idx))[::-1]
            life[idx] = len(idx) + X.shape[1] - 1          # flights in this engine's life
        self.remaining, self.life = remaining, life
        self.pool = np.where(remaining >= dt_min)[0]

    def next(self):
        i = self.rng.choice(self.pool, self.batch)
        dt = np.minimum(self.rng.integers(self.dt_min, self.dt_max + 1, self.batch), self.remaining[i])
        j = i + dt
        return (tf.constant(np.concatenate([self.X[i], self.X[j]])),
                tf.constant(np.concatenate([self.U[i], self.U[j]])),
                tf.constant((dt / self.life[i]).astype(np.float32)),
                tf.constant(np.concatenate([self.y[i], self.y[j]])))


class RegimeKoopmanTrainer(EnhancedKePINTrainer):

    def __init__(self, *args, sampler=None, regime_loss="mono", regime_weight=1.0,
                 margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        if regime_loss not in ("mono", "sup"):
            raise ValueError(regime_loss)
        self.sampler = sampler
        self.regime_loss_kind = regime_loss
        self.regime_weight = float(regime_weight)
        self.margin = float(margin)
        self.regime_log = []          # (regime loss, mean h drop) per training step

    def regime_loss(self, Xp, Up, frac, yp):
        h = tf.reshape(self.model.regime_health((Xp, Up), training=True), [-1])
        n = tf.shape(frac)[0]
        h_early, h_late = h[:n], h[n:]
        if self.regime_loss_kind == "mono":
            loss = tf.reduce_mean(tf.nn.relu(h_late - h_early + self.margin * frac))
        else:
            loss = tf.reduce_mean(tf.square(h - yp))
        return loss, tf.reduce_mean(h_early - h_late)

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

    def _with_regime(self, inputs, Y, Xp, Up, frac, yp):
        with tf.GradientTape() as tape:
            rul_pred, koopman_out = self.model(inputs, training=True)
            rul_pred_f32 = tf.cast(rul_pred, tf.float32)
            total_loss, loss_dict = self.loss_fn(tf.cast(Y, tf.float32), rul_pred_f32,
                                                 self._f32(koopman_out))
            lr, drop = self.regime_loss(Xp, Up, frac, yp)
            total_loss = total_loss + self.regime_weight * lr
        loss_dict["regime"] = lr
        loss_dict["regime_h_drop"] = drop          # mean h(t) - h(t+dt): should become > 0
        g, s = self._apply(tape, total_loss)
        return total_loss, loss_dict, rul_pred_f32, g, s

    @tf.function(jit_compile=False)
    def _step(self, X_batch, Y_batch, Xp, Up, frac, yp):
        return self._with_regime(X_batch, Y_batch, Xp, Up, frac, yp)

    @tf.function(jit_compile=False)
    def _step_mixup(self, X1, Y1, X2, Y2, lam, Xp, Up, frac, yp):
        lam_y = tf.reshape(lam[:, 0, 0], (-1, 1))
        Y_mix = lam_y * Y1 + (1.0 - lam_y) * Y2
        inputs_mix = (lam * X1[0] + (1.0 - lam) * X2[0], lam_y * X1[1] + (1.0 - lam_y) * X2[1])
        return self._with_regime(inputs_mix, Y_mix, Xp, Up, frac, yp)

    # the parent fit loop calls these two; a fresh pair batch is drawn every step.
    # The parent history ignores extra losses, so they are logged here per step.
    def _log(self, out):
        self.regime_log.append((float(out[1]["regime"]), float(out[1]["regime_h_drop"])))
        return out

    def train_step(self, X_batch, Y_batch):
        return self._log(self._step(X_batch, Y_batch, *self.sampler.next()))

    def train_step_mixup(self, X1, Y1, X2, Y2, lam):
        return self._log(self._step_mixup(X1, Y1, X2, Y2, lam, *self.sampler.next()))
