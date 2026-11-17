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
         With health_map="ordered" the shift can only speed decay up as health falls
         (healthy -> slow forgetting, near failure -> fast forgetting):
             s_k(h) = s_k - gap_k * (1 - h),   gap_k = 2 * sigmoid(c_k) in (0, 2)
         h = 1 gives the "healthy" operator s_k; h = 0 the "failure" operator s_k - gap_k.
         gap_k starts small (c_k = -4 -> 0.036) and is learned; gap ~ 0 means the data
         did not support regime-dependent decay.
both   : exo + health; the operating condition drives rho and theta per step, health
         shifts the decay on top.
hyper  : hypernetwork Koopman (after Hyper-LR-PINN, Cho et al., NeurIPS 2023: shared basis,
         context-generated diagonal). The basis P stays shared and orthogonal; a hypernetwork
         reads a context c = [mean_t z, z_T] (+ an external signal, e.g. the engine's age) and
         outputs per-mode shifts of the decay logit and the rotation:
             rho_k(c) = sigmoid(s_k + ds_k(c)) * g_k(c),   theta_k(c) = theta_k + dth_k(c)
         ds, dth are bounded (2 tanh, 0.5 tanh) and zero-initialised (starts as the static K).
         With hyper_rank=True, g_k(c) = clip(ReLU(1 + dg_k(c)), 0, 1) can switch a mode OFF for
         a given context (adaptive Koopman rank, as the ReLU on the singular values in the
         paper); otherwise g_k = 1. The operator is then fixed within the window, per engine.

adapt (how the operator adapts per engine, used with mode="health"):
  gate    : the health-gated decay above.
  mixture : R = 3 regime operators (healthy, degrading, near failure) sharing the basis P,
            each with its own decay/rotation (s + o_r, theta + d_r). Soft regime probabilities
            pi = softmax(MLP([mean z, z_T])) blend them. A sum of scaled rotations is a scaled
            rotation, so the blend stays a block rotation-scaling:
                A = sum_r pi_r rho_r cos(theta_r),  B = sum_r pi_r rho_r sin(theta_r)
                rho = |A + iB|,  theta = arg(A + iB)
            The health index is h = pi . [1, 0.5, 0], so the regime loss orders the regimes.
            Offsets start at o = [0, -0.5, -1] (later regimes forget faster) and are learned.
  engine  : per-engine eigenvalues. In the shared basis, each mode's eigenvalue is fitted to
            this engine's own window in closed form (1-D DMD per mode):
                lam_hat = sum_t c_{t+1} conj(c_t) / sum_t |c_t|^2,   c_t = a_t + i b_t
            and blended with the fleet eigenvalue in log-polar form with a learned weight
            alpha_k = sigmoid(.) (starts at 0.5). The Koopman losses keep using the fleet
            operator (otherwise the in-window fit would make them trivial); the per-engine
            eigenvalues reach the RUL head as spectral features and through the forecast.

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
                 shift_bound=1.0, theta_bound=0.5, cond_units=32, s_init="zeros",
                 health_map="free", gap_init=-4.0, external_health=False, adapt="gate",
                 n_regimes=3, hyper_ctx_dim=0, hyper_rank=False, hyper_use_window=True, **kwargs):
        super().__init__(**kwargs)
        if latent_dim % 2:
            raise ValueError("latent_dim must be even (modes come in conjugate pairs)")
        if mode not in ("static", "exo", "health", "both", "hyper"):
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
        if health_map not in ("free", "ordered"):
            raise ValueError(f"unknown health_map {health_map}")
        self.health_map = health_map
        self.gap_init = float(gap_init)          # gap_k = 2*sigmoid(gap_init) at the start
        self.external_health = external_health   # h is given by the caller (e.g. Signal A)
        if adapt not in ("gate", "mixture", "engine"):
            raise ValueError(f"unknown adapt {adapt}")
        if adapt != "gate" and (mode != "health" or external_health):
            raise ValueError("adapt='mixture'/'engine' need mode='health' and the learned health head")
        self.adapt = adapt
        self.n_regimes = n_regimes
        self.hyper_ctx_dim = int(hyper_ctx_dim)      # size of the external context (0 = window only)
        self.hyper_rank = bool(hyper_rank)
        self.hyper_use_window = bool(hyper_use_window)   # False: context = external signal only (bottleneck)

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
        if self.uses_health and not self.external_health:
            # health head reads [mean_t z, z_T]; a = per-mode sensitivity, zero-initialised
            self.health_hidden = keras.layers.Dense(self.cond_units, activation="relu", name="health_dense1")
            self.health_hidden.build((None, 2 * d))
            if self.adapt == "mixture":
                self.regime_out = keras.layers.Dense(self.n_regimes, name="regime_out")
                self.regime_out.build((None, self.cond_units))
            else:
                self.health_out = keras.layers.Dense(1, activation="sigmoid", name="health_out")
                self.health_out.build((None, self.cond_units))
        if self.uses_health and self.adapt == "mixture":
            R = self.n_regimes
            offsets = np.repeat(-np.linspace(0, 1, R)[:, None], m, 1).astype(np.float32)
            self.reg_ds = self.add_weight(name="regime_decay_offset", shape=(R, m),
                                          initializer=lambda shape, dtype=None: tf.constant(offsets),
                                          trainable=True)
            self.reg_dth = self.add_weight(name="regime_theta_offset", shape=(R, m),
                                           initializer="zeros", trainable=True)
        elif self.adapt == "engine":
            self.engine_mix_logit = self.add_weight(name="engine_mix_logit", shape=(m,),
                                                    initializer="zeros", trainable=True)
        elif self.uses_health:
            if self.health_map == "ordered":
                self.gap_logit = self.add_weight(name="health_gap_logit", shape=(m,),
                                                 initializer=keras.initializers.Constant(self.gap_init),
                                                 trainable=True)
            else:
                self.a = self.add_weight(name="health_sensitivity", shape=(m,),
                                         initializer=keras.initializers.Zeros(), trainable=True)
        if self.mode == "hyper":
            n_out = 3 * m if self.hyper_rank else 2 * m
            self.hyper_hidden = keras.layers.Dense(64, activation="relu", name="hyper_hidden")
            self.hyper_out = keras.layers.Dense(n_out, kernel_initializer="zeros", bias_initializer="zeros",
                                                name="hyper_out")
            self.hyper_hidden.build((None, (2 * d if self.hyper_use_window else 0) + self.hyper_ctx_dim))
            self.hyper_out.build((None, 64))
        super().build(input_shape)

    @property
    def uses_exo(self):
        return self.mode in ("exo", "both")

    @property
    def uses_health(self):
        return self.mode in ("health", "both") and self.adapt != "engine"

    def _health_feats(self, z):
        z = tf.cast(z, tf.float32)
        return self.health_hidden(tf.concat([tf.reduce_mean(z, axis=1), z[:, -1]], axis=-1))

    def regime_probs(self, z):
        """Mixture: soft regime probabilities (batch, R), ordered healthy -> near failure."""
        return tf.nn.softmax(self.regime_out(self._health_feats(z)), axis=-1)

    def health(self, z):
        """Scalar health index per window, (batch, 1) in (0, 1)."""
        if self.adapt == "mixture":
            levels = tf.linspace(1.0, 0.0, self.n_regimes)[:, None]          # [1, 0.5, 0]
            return tf.matmul(self.regime_probs(z), levels)
        return self.health_out(self._health_feats(z))

    def hyper_params(self, z, signal=None):
        """Hypernetwork: context -> per-engine (log_rho, theta) (B, m) and mode gates (or None)."""
        m = self.n_modes
        z = tf.cast(z, tf.float32)
        ctx = [tf.reduce_mean(z, axis=1), z[:, -1]] if self.hyper_use_window else []
        if self.hyper_ctx_dim:
            ctx.append(tf.reshape(tf.cast(signal, tf.float32), [-1, self.hyper_ctx_dim]))
        out = self.hyper_out(self.hyper_hidden(tf.concat(ctx, -1)))
        ds = 2.0 * tf.tanh(out[:, :m])
        dth = 0.5 * tf.tanh(out[:, m:2 * m])
        log_rho = tf.math.log_sigmoid(self.s + ds)
        g = None
        if self.hyper_rank:
            g = tf.clip_by_value(tf.nn.relu(1.0 + out[:, 2 * m:]), 0.0, 1.0)
            log_rho = log_rho + tf.math.log(g + 1e-6)                        # g = 0 -> mode off
        return log_rho, self.theta + dth, g

    def mixture_params(self, z):
        """Blend of the R regime operators -> per-window (log_rho, theta), each (B, m)."""
        pi = self.regime_probs(z)                                            # (B, R)
        rho_r = tf.sigmoid(self.s[None] + self.reg_ds)                       # (R, m)
        th_r = self.theta[None] + self.reg_dth
        A = tf.matmul(pi, rho_r * tf.cos(th_r))
        B = tf.matmul(pi, rho_r * tf.sin(th_r))
        return 0.5 * tf.math.log(A ** 2 + B ** 2 + 1e-12), tf.atan2(B, A), pi

    def engine_params(self, zt):
        """Per-engine eigenvalues: closed-form fit of each mode on the window, blended with
        the fleet eigenvalue. zt: modal coordinates (B, T, d). Returns (log_rho, theta, alpha)."""
        m = self.n_modes
        a, b = zt[..., :m], zt[..., m:]
        a0, b0, a1, b1 = a[:, :-1], b[:, :-1], a[:, 1:], b[:, 1:]
        den = tf.reduce_sum(a0 ** 2 + b0 ** 2, 1) + 1e-6
        re = tf.reduce_sum(a1 * a0 + b1 * b0, 1) / den
        im = tf.reduce_sum(b1 * a0 - a1 * b0, 1) / den
        lr_hat = tf.clip_by_value(0.5 * tf.math.log(re ** 2 + im ** 2 + 1e-12), np.log(1e-3), -1e-3)
        th_hat = tf.atan2(im, re)
        lr_g, th_g = tf.math.log_sigmoid(self.s), self.theta
        dth = tf.atan2(tf.sin(th_hat - th_g), tf.cos(th_hat - th_g))       # wrapped difference
        alpha = tf.sigmoid(self.engine_mix_logit)
        return lr_g + alpha * (lr_hat - lr_g), th_g + alpha * dth, alpha

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
            ds = ds + self.health_shift(h)[:, None, :]                                    # (B, 1, m)
        log_rho = tf.math.log_sigmoid(self.s + ds)
        theta = self.theta + dth
        shape = tf.shape(log_rho) if use_exo else tf.stack([tf.shape(h)[0], T, self.n_modes])
        return tf.broadcast_to(log_rho, shape), tf.broadcast_to(theta, shape)

    def health_gap(self):
        """Ordered map: decay-logit drop of each mode between healthy (h=1) and failed (h=0)."""
        return 2.0 * tf.sigmoid(self.gap_logit)

    def health_shift(self, h):
        """Shift of the decay logits s_k for health h (batch, 1) -> (batch, m)."""
        if self.health_map == "ordered":
            return -self.health_gap() * (1.0 - h)
        return self.shift_bound * tf.tanh(self.a * (2.0 * h - 1.0))

    def propagate(self, z_last, log_rho, theta, steps):
        """Carry the current state `steps` flights ahead with this engine's operator.

        log_rho, theta: per-engine mode parameters (B, m) from call() ("adapt_log_rho",
        "adapt_theta"). Returns the propagated state in modal coordinates, (B, d). This is
        how the adaptive operator reaches the RUL head.
        """
        m = self.n_modes
        zt = tf.matmul(tf.cast(z_last, tf.float32), self.basis())
        a, b = self._apply(zt[:, :m], zt[:, m:], steps * log_rho, steps * theta)
        return tf.concat([a, b], -1)

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

    def health_rollout(self, z_last, horizon, readout_w, readout_b, log_rho=None, theta=None):
        """Project the current state forward with the (static) operator and read health.

        z_last: (B, d) current latent state; readout_w: (d,) health readout in modal
        coordinates; readout_b: scalar. Returns health H_k for k = 0..horizon, (B, horizon+1).
        """
        m = self.n_modes
        zt = tf.matmul(tf.cast(z_last, tf.float32), self.basis())           # (B, d)
        a, b = zt[:, None, :m], zt[:, None, m:]                              # (B, 1, m)
        k = tf.range(horizon + 1, dtype=tf.float32)[:, None]                 # (H+1, 1)
        if log_rho is None:                                                  # fixed operator
            log_rho, theta = self.mode_params()
        else:                                                                # this engine's operator
            log_rho, theta = log_rho[:, None, :], theta[:, None, :]          # (B, 1, m)
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
    def call(self, z_sequence, signal=None, h=None, training=None):
        z = tf.cast(z_sequence, tf.float32)
        T = z.shape[1]
        m = self.n_modes
        zt = tf.matmul(z, self.basis())                  # modal coordinates (B, T, d)
        a, b = zt[..., :m], zt[..., m:]
        if self.uses_health and h is None:
            h = self.health(z)
        h = tf.cast(h, tf.float32) if self.uses_health else None
        extra = {}
        if self.mode == "hyper":
            lr_s, th_s, g = self.hyper_params(z, signal)                    # (B, m)
            shape = tf.stack([tf.shape(z)[0], T, m])
            log_rho = tf.broadcast_to(lr_s[:, None], shape)
            theta = tf.broadcast_to(th_s[:, None], shape)
            if g is not None:
                extra["hyper_mode_gate"] = g
        elif self.adapt == "mixture":
            lr_s, th_s, pi = self.mixture_params(z)                         # (B, m)
            shape = tf.stack([tf.shape(z)[0], T, m])
            log_rho = tf.broadcast_to(lr_s[:, None], shape)
            theta = tf.broadcast_to(th_s[:, None], shape)
            extra["regime_probs"] = pi
        elif self.adapt == "engine":
            log_rho, theta = self.mode_params()                              # fleet operator: losses
            lr_s, th_s, _ = self.engine_params(zt)
        else:
            log_rho, theta = self.mode_params(signal, T, h)
            if len(log_rho.shape) == 3:
                lr_s, th_s = log_rho[:, -1], theta[:, -1]
            else:
                shape = tf.stack([tf.shape(z)[0], m])
                lr_s, th_s = tf.broadcast_to(log_rho, shape), tf.broadcast_to(theta, shape)
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

        if dynamic or self.adapt == "engine":   # this engine's operator (last step)
            rho_T = tf.exp(lr_s)
            lam = tf.complex(rho_T * tf.cos(th_s), rho_T * tf.sin(th_s))
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
            "adapt_log_rho": lr_s,
            "adapt_theta": th_s,
            **extra,
        }
        if h is not None:
            out["health"] = h
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(latent_dim=self.latent_dim, rollout_steps=self.rollout_steps,
                        mode=self.mode, exo_dim=self.exo_dim, shift_bound=self.shift_bound,
                        theta_bound=self.theta_bound, cond_units=self.cond_units,
                        s_init=self.s_init, health_map=self.health_map,
                        gap_init=self.gap_init, external_health=self.external_health,
                        adapt=self.adapt, n_regimes=self.n_regimes,
                        hyper_ctx_dim=self.hyper_ctx_dim, hyper_rank=self.hyper_rank,
                        hyper_use_window=self.hyper_use_window))
        return cfg
