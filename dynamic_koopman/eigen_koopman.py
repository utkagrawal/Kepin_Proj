# -*- coding: utf-8 -*-
"""
Eigen-parameterised (dynamic) Koopman operator.

Instead of K = U diag(sigmoid(s)) V^T followed by a numerical eigendecomposition,
K is built directly from its eigenvalues:

    K = P · blockdiag( rho_k · R(theta_k) ) · P^T        k = 1 .. d/2
    P        orthogonal (Cayley transform of a free matrix)
    rho_k    = sigmoid(s_k)       in (0, 1)   -> decay of mode k
    theta_k  = free angle                      -> oscillation frequency of mode k
    R(theta) = 2x2 rotation

so the eigenvalues are exactly rho_k · exp(±i theta_k) and no eigvals call is needed.
Because P is orthogonal, the Koopman one-step / multi-step MSE is identical in
modal coordinates z~ = z P, which is where all propagation happens (O(d) per step).

Modes
-----
static : rho, theta are constants (drop-in replacement for the original layer).
exo    : rho_t, theta_t depend on an exogenous per-time-step input u_t (e.g. the
         operating condition):  s_t = s + bound*tanh(f(u_t)),  theta_t = theta + bound_theta*tanh(g(u_t)).
         The operator then changes at every time step. All blocks share the basis P,
         so products of operators stay block rotations-scalings:
             prod_j rho_j  and  sum_j theta_j   (closed form, no matrix powers).
health : decay depends on a learned scalar health index h in (0, 1) per window, read
         from the latent sequence:  s_k = s_k + bound*tanh(a_k * (2h - 1)).
         a_k (per-mode sensitivity to health) starts at 0 -> starts as the static operator.
both   : exo + health; the operating condition drives rho and theta per step, health
         shifts the decay on top.

The `signal` argument for mode="exo" is the flattened per-step input (batch, T*exo_dim);
it is flattened so the existing mixup code (which mixes 2-D signals) works unchanged.
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable


@register_keras_serializable(package="DynamicKoopman")
class EigenKoopmanOperator(keras.layers.Layer):

    def __init__(self, latent_dim, rollout_steps=3, mode="static", exo_dim=0,
                 shift_bound=1.0, theta_bound=0.5, cond_units=32, s_init="zeros", **kwargs):
        super().__init__(**kwargs)
        if latent_dim % 2:
            raise ValueError("latent_dim must be even (modes come in conjugate pairs)")
        if mode not in ("static", "exo", "health", "both"):
            raise ValueError(f"unknown mode {mode}")
        self.latent_dim = latent_dim
        self.n_modes = latent_dim // 2
        self.rollout_steps = rollout_steps
        self.mode = mode
        self.exo_dim = exo_dim
        self.shift_bound = shift_bound
        self.theta_bound = theta_bound
        self.cond_units = cond_units
        self.s_init = s_init

    def build(self, input_shape):
        d, m = self.latent_dim, self.n_modes
        self.M_P = self.add_weight(
            name="M_P", shape=(d, d),
            initializer=keras.initializers.RandomNormal(stddev=0.05), trainable=True)
        # "spread": decays from 0.5 to 0.993 per step, so some modes can carry
        # information over long horizons (needed by the time-to-failure head)
        s_init = (keras.initializers.Zeros() if self.s_init == "zeros"
                  else keras.initializers.RandomUniform(0.0, 5.0))
        self.s = self.add_weight(name="s", shape=(m,), initializer=s_init, trainable=True)
        self.theta = self.add_weight(
            name="theta", shape=(m,),
            initializer=keras.initializers.RandomUniform(0.0, np.pi), trainable=True)
        if self.uses_exo:
            # zero-initialised output -> starts exactly as the static operator
            self.cond_hidden = keras.layers.Dense(self.cond_units, activation="relu", name="exo_dense1")
            self.cond_out = keras.layers.Dense(2 * m, kernel_initializer="zeros",
                                               bias_initializer="zeros", name="exo_out")
            self.cond_hidden.build((None, self.exo_dim))
            self.cond_out.build((None, self.cond_units))
        if self.uses_health:
            # health head reads [mean_t z, z_T]; a = per-mode sensitivity, zero-initialised
            self.health_hidden = keras.layers.Dense(self.cond_units, activation="relu", name="health_dense1")
            self.health_out = keras.layers.Dense(1, activation="sigmoid", name="health_out")
            self.health_hidden.build((None, 2 * d))
            self.health_out.build((None, self.cond_units))
            self.a = self.add_weight(name="health_sensitivity", shape=(m,),
                                     initializer=keras.initializers.Zeros(), trainable=True)
        super().build(input_shape)

    @property
    def uses_exo(self):
        return self.mode in ("exo", "both")

    @property
    def uses_health(self):
        return self.mode in ("health", "both")

    def health(self, z):
        """Scalar health index per window, (batch, 1) in (0, 1)."""
        feats = tf.concat([tf.reduce_mean(z, axis=1), z[:, -1]], axis=-1)
        return self.health_out(self.health_hidden(feats))

    # ------------------------------------------------------------------
    def basis(self):
        d = self.latent_dim
        I = tf.eye(d, dtype=tf.float32)
        A = self.M_P - tf.transpose(self.M_P)
        return tf.linalg.solve(I + A, I - A)

    def mode_params(self, signal=None, T=None, h=None):
        """Returns log_rho, theta with shape (m,) [static] or (batch, T, m) [dynamic].

        signal: flattened per-step exogenous input (exo / both); h: (batch, 1) health (health / both).
        """
        use_exo = self.uses_exo and signal is not None
        use_health = self.uses_health and h is not None
        if not (use_exo or use_health):
            return tf.math.log_sigmoid(self.s), self.theta
        ds, dth = 0.0, 0.0
        if use_exo:
            u = tf.reshape(tf.cast(signal, tf.float32), [-1, T, self.exo_dim])
            out = self.cond_out(self.cond_hidden(u))                  # (B, T, 2m)
            ds_e, dth_e = tf.split(out, 2, axis=-1)
            ds = self.shift_bound * tf.tanh(ds_e)
            dth = self.theta_bound * tf.tanh(dth_e)
        if use_health:
            ds = ds + self.shift_bound * tf.tanh(self.a * (2.0 * h - 1.0))[:, None, :]   # (B, 1, m)
        log_rho = tf.math.log_sigmoid(self.s + ds)
        theta = self.theta + dth
        shape = tf.shape(log_rho) if use_exo else tf.stack([tf.shape(h)[0], T, self.n_modes])
        return tf.broadcast_to(log_rho, shape), tf.broadcast_to(theta, shape)

    @staticmethod
    def _apply(a, b, log_rho, theta):
        rho = tf.exp(log_rho)
        c, s = rho * tf.cos(theta), rho * tf.sin(theta)
        return c * a - s * b, s * a + c * b

    def eigenvalues(self, signal=None, T=None, h=None):
        log_rho, theta = self.mode_params(signal, T, h)
        if len(log_rho.shape) == 3:              # dynamic: operator at the last step
            log_rho, theta = log_rho[:, -1], theta[:, -1]
        rho = tf.exp(log_rho)
        lam = tf.complex(rho * tf.cos(theta), rho * tf.sin(theta))
        return tf.concat([lam, tf.math.conj(lam)], axis=-1)

    def health_rollout(self, z_last, horizon, readout_w, readout_b):
        """Project the current state forward with the (static) operator and read health.

        z_last: (B, d) current latent state; readout_w: (d,) health readout in modal
        coordinates; readout_b: scalar. Returns health H_k for k = 0..horizon, (B, horizon+1).
        """
        m = self.n_modes
        zt = tf.matmul(tf.cast(z_last, tf.float32), self.basis())           # (B, d)
        a, b = zt[:, None, :m], zt[:, None, m:]                              # (B, 1, m)
        log_rho, theta = self.mode_params()
        k = tf.range(horizon + 1, dtype=tf.float32)[:, None]                 # (H+1, 1)
        ak, bk = self._apply(a, b, k * log_rho, k * theta)                   # (B, H+1, m)
        return tf.tensordot(ak, readout_w[:m], 1) + tf.tensordot(bk, readout_w[m:], 1) + readout_b

    def koopman_matrix(self):
        """Dense K of the static part (for analysis only, never used in training)."""
        m = self.n_modes
        log_rho, theta = self.mode_params()
        rho = tf.exp(log_rho)
        c, s = rho * tf.cos(theta), rho * tf.sin(theta)
        B = tf.concat([tf.concat([tf.linalg.diag(c), -tf.linalg.diag(s)], 1),
                       tf.concat([tf.linalg.diag(s), tf.linalg.diag(c)], 1)], 0)
        P = self.basis()
        # row-vector convention: z~ = z P, z~_{t+1} = z~_t B^T  ->  z_{t+1} = z_t P B^T P^T
        return tf.transpose(P @ tf.transpose(B) @ tf.transpose(P))

    # ------------------------------------------------------------------
    def call(self, z_sequence, signal=None, training=None):
        z = tf.cast(z_sequence, tf.float32)
        T = z.shape[1]
        m = self.n_modes
        zt = tf.matmul(z, self.basis())                  # modal coordinates (B, T, d)
        a, b = zt[..., :m], zt[..., m:]
        h = self.health(z) if self.uses_health else None
        log_rho, theta = self.mode_params(signal, T, h)
        dynamic = len(log_rho.shape) == 3

        # one-step: z~(t+1) = B_t z~(t)
        lr1 = log_rho[:, :-1] if dynamic else log_rho
        th1 = theta[:, :-1] if dynamic else theta
        a1, b1 = self._apply(a[:, :-1], b[:, :-1], lr1, th1)
        one_pred = tf.concat([a1, b1], -1)
        one_tgt = zt[:, 1:]

        # multi-step: z~(t+k) = (B_{t+k-1} ... B_t) z~(t), closed form via cumulative sums
        H = min(self.rollout_steps, 5)
        n_start = T - H                                  # starts t = 0 .. T-H-1
        a0, b0 = a[:, :n_start], b[:, :n_start]
        preds, tgts = [], []
        if dynamic:
            cs_lr = tf.pad(tf.cumsum(log_rho, axis=1), [[0, 0], [1, 0], [0, 0]])
            cs_th = tf.pad(tf.cumsum(theta, axis=1), [[0, 0], [1, 0], [0, 0]])
        for k in range(1, H + 1):
            if dynamic:
                lrk = cs_lr[:, k:k + n_start] - cs_lr[:, :n_start]
                thk = cs_th[:, k:k + n_start] - cs_th[:, :n_start]
            else:
                lrk, thk = k * log_rho, k * theta
            ak, bk = self._apply(a0, b0, lrk, thk)
            preds.append(tf.concat([ak, bk], -1))
            tgts.append(zt[:, k:k + n_start])

        if dynamic:   # operator at the last step of the window
            rho_T = tf.exp(log_rho[:, -1])
            lam = tf.complex(rho_T * tf.cos(theta[:, -1]), rho_T * tf.sin(theta[:, -1]))
            eig = tf.concat([lam, tf.math.conj(lam)], axis=-1)
        else:
            eig = self.eigenvalues()
        out = {
            "one_step_pred": one_pred,
            "one_step_target": one_tgt,
            "multi_step_pred": tf.stack(preds, axis=2),
            "multi_step_target": tf.stack(tgts, axis=2),
            "eigenvalues": eig,
            "final_state": z[:, -1, :],
        }
        if h is not None:
            out["health"] = h
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(latent_dim=self.latent_dim, rollout_steps=self.rollout_steps,
                        mode=self.mode, exo_dim=self.exo_dim, shift_bound=self.shift_bound,
                        theta_bound=self.theta_bound, cond_units=self.cond_units,
                        s_init=self.s_init))
        return cfg
