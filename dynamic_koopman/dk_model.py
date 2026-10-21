# -*- coding: utf-8 -*-
"""KePIN with a swappable Koopman layer and prediction head.

koopman="svd" keeps the original KePIN layer (K = U diag(sigmoid(s)) V^T + eigvals).
koopman="eig" uses EigenKoopmanOperator (no eigendecomposition; static/exo/health/both).
koopman="none" removes Koopman completely (ablation): no Koopman layer, no spectral features
               into the RUL head, and (in train.py) no Koopman losses. Same encoder and head.

head="kepin" : original KePIN RUL head (pooled latent + spectral features).
head="ttf"   : Koopman time-to-failure. A linear readout gives a health score of any
               latent state; the operator projects the current state forward k = 1..H
               steps; RUL = soft count of steps before the projected health crosses 0:
                   S_k = prod_{j<=k} sigmoid(-H_j / tau),   RUL = sum_k S_k   (in [0, H])
               The prediction *is* the Koopman rollout, so it is explained by the
               projected health trajectory of each engine.
head="wear"  : TTF plus a wear accumulator. A stable operator can only make modes fade,
               so the plain TTF trajectories never reached the threshold. Here the state
               is augmented with an integrator mode (eigenvalue exactly 1) driven by a
               learned per-engine wear rate v >= 0:
                   H_k = H_koopman(k) + v * k            (projected health, failure at 0)
                   g_k = H_k / v                          (same curve in units of flights)
                   S_k = prod_{j<=k} sigmoid(-g_j / tau),  RUL = sum_k S_k,  tau = 2 flights
               Measuring the crossing in flights means survival only drops near the actual
               crossing, so RUL ~ the step where projected health reaches 0
               (~ -current health / wear rate). Per engine we save the current health,
               the wear rate and the projected trajectory.

Condition-aware Koopman (cond_norm / koop_control; inputs = (raw window, condition one-hot per step)):
  cond_norm    : learned per-operating-condition scale and offset on the raw (unsmoothed) sensors,
                 x' = x * gamma[c_t] + beta[c_t], initialised to per-condition standardisation.
                 KePIN's EMA smoothing is moved inside the model as a fixed layer so it runs
                 AFTER this step (smoothing first would blend different conditions).
  koop_control : Koopman with control input. The latent is split as z_t = s_t + C u_t, where
                 C u_t is the flight-condition offset and s_t the condition-free state that must
                 follow the Koopman rule (s_{t+1} = K s_t, i.e. z_{t+1} = K z_t + C u_{t+1} - K C u_t).
                 KePIN's Koopman losses and the RUL head both use s instead of z.
With gamma = 1, beta = 0 and C = 0 this is exactly the original KePIN (checked in tests).
Everything else (encoder, losses, trainer) is the original KePIN code.
"""
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIR, "kepin_core"))

import numpy as np
import tensorflow as tf
import keras

from kepin_model import KePINModel
from koopman_module import extract_spectral_features
from eigen_koopman import EigenKoopmanOperator

TTF_HORIZON = 125          # = RUL cap of the C-MAPSS labels
WEAR_TAU = 2.0             # width of the soft threshold crossing, in flights


class DynamicKePIN(KePINModel):

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, n_active_losses=7,
                 koopman="eig", koopman_mode="static", exo_dim=0, head="kepin",
                 cond_norm=False, koop_control=False, cond_stats=None, ema_alpha=None, **kwargs):
        super().__init__(input_shape_tuple, arch_config=arch_config, n_train=n_train,
                         n_active_losses=n_active_losses, **kwargs)
        self.koopman_kind = koopman
        self.koopman_mode = koopman_mode
        self.head = head
        if head in ("ttf", "wear") and (koopman != "eig" or koopman_mode != "static"):
            raise ValueError("the time-to-failure head uses the static eig operator")
        if koopman == "eig":
            self.koopman = EigenKoopmanOperator(
                latent_dim=self.arch_config["latent_dim"],
                rollout_steps=self.arch_config["rollout"],
                mode=koopman_mode, exo_dim=exo_dim,
                s_init="spread" if head in ("ttf", "wear") else "zeros", name="eigen_koopman")
        elif koopman == "none":
            if head != "kepin" or koopman_mode != "static":
                raise ValueError("koopman='none' is the plain KePIN-head ablation")
            self.koopman = None
        elif koopman != "svd":
            raise ValueError(koopman)
        self.cond_norm, self.koop_control = cond_norm, koop_control
        self.condition_aware = cond_norm or koop_control
        if self.condition_aware:
            if koopman != "svd" or head != "kepin" or koopman_mode != "static":
                raise ValueError("condition-aware Koopman is built on the original KePIN (svd, kepin head)")
            seq_len, n_feat = input_shape_tuple
            self.n_cond = exo_dim
            t = np.arange(seq_len)
            a = float(ema_alpha)
            M = np.where(t[None, :] <= t[:, None], a * (1 - a) ** (t[:, None] - t[None, :]), 0.0)
            M[:, 0] = (1 - a) ** t                      # y_0 = x_0 (lfilter_zi start, as in KePIN)
            self.ema_matrix = tf.constant(M, tf.float32)  # (T, T): y = M x along time
            if cond_norm:
                mu, sd = cond_stats                     # (n_cond, n_feat) on raw training windows
                self.cn_gamma = self.add_weight(name="condnorm_gamma", shape=(exo_dim, n_feat),
                                                initializer="ones")
                self.cn_beta = self.add_weight(name="condnorm_beta", shape=(exo_dim, n_feat),
                                               initializer="zeros")
                self.cn_gamma.assign((1.0 / sd).astype(np.float32))
                self.cn_beta.assign((-mu / sd).astype(np.float32))
            if koop_control:
                self.control_C = self.add_weight(name="koopman_control_C",
                                                 shape=(exo_dim, self.arch_config["latent_dim"]),
                                                 initializer=keras.initializers.Zeros())
        if head in ("ttf", "wear"):
            d = self.arch_config["latent_dim"]
            self.health_w = self.add_weight(name="ttf_health_w", shape=(d,),
                                            initializer=keras.initializers.RandomNormal(stddev=0.05))
            # bias -5: per-step survival sigmoid(5) ~ 0.993 -> initial predictions mid-range
            self.health_b = self.add_weight(name="ttf_health_b", shape=(),
                                            initializer=keras.initializers.Constant(-5.0))
        if head == "wear":
            # wear rate v = softplus(w_v . z_T + b_v); init v ~ 0.08 -> first crossings ~60 flights
            self.wear_w = self.add_weight(name="wear_rate_w", shape=(d,),
                                          initializer=keras.initializers.RandomNormal(stddev=0.01))
            self.wear_b = self.add_weight(name="wear_rate_b", shape=(),
                                          initializer=keras.initializers.Constant(-2.48))

    def encode(self, x, training=None):
        """Encoder of KePINModel.call, unchanged: input window -> latent sequence."""
        x = self.input_proj(x)
        x = self.input_bn(x, training=training)
        x = tf.nn.relu(x)
        for block in self.encoder_blocks:
            residual = x
            h = block["conv1"](x)
            h = block["bn1"](h, training=training)
            h = tf.nn.relu(h)
            h = block["conv2"](h)
            h = block["bn2"](h, training=training)
            se = block["se_gap"](h)
            se = block["se_dense1"](se)
            se = block["se_dense2"](se)
            se = block["se_reshape"](se)
            h = block["se_mul"]([h, se])
            if block["skip_proj"] is not None:
                residual = block["skip_proj"](residual)
                residual = block["skip_bn"](residual, training=training)
            x = tf.nn.relu(h + residual)
        x = self.bilstm(x, training=training)
        x = self.lstm_ln(x, training=training)
        x = self.post_lstm_proj(x)
        x = self.post_lstm_bn(x, training=training)
        attn_out = self.mha(x, x, training=training)
        attn_out = self.mha_dropout(attn_out, training=training)
        x = self.mha_ln(x + attn_out, training=training)
        ff_out = self.ff_dense1(x)
        ff_out = self.ff_dense2(ff_out)
        ff_out = self.ff_dropout(ff_out, training=training)
        x = self.ff_ln(x + ff_out, training=training)
        z_seq = self.latent_proj(x)
        return self.latent_bn(z_seq, training=training)

    def _condition_aware_call(self, inputs, training):
        x, u_flat = inputs
        x = tf.cast(x, tf.float32)
        u = tf.reshape(tf.cast(u_flat, tf.float32), [-1, x.shape[1], self.n_cond])   # (B, T, C)
        if self.cond_norm:
            x = x * tf.einsum("btc,cf->btf", u, self.cn_gamma) + tf.einsum("btc,cf->btf", u, self.cn_beta)
        x = tf.einsum("ts,bsf->btf", self.ema_matrix, x)          # KePIN's EMA, after the condition step
        z_seq = tf.cast(self.encode(x, training=training), tf.float32)
        if self.koop_control:
            z_seq = z_seq - tf.einsum("btc,cd->btd", u, self.control_C)   # condition-free state s
        koopman_out = self.koopman(z_seq, signal=None, training=training)
        pooled = tf.concat([self.gap(z_seq), self.gmp(z_seq)], axis=-1)
        spec = extract_spectral_features(koopman_out["eigenvalues"], top_k=self.spectral_k)
        spec = tf.tile(tf.expand_dims(tf.cast(spec, tf.float32), 0), [tf.shape(x)[0], 1])
        h = self.head_dense1(tf.concat([pooled, spec], axis=-1))
        h = self.head_bn1(h, training=training)
        h = self.head_drop1(h, training=training)
        h = self.head_dense2(h)
        h = self.head_bn2(h, training=training)
        h = self.head_drop2(h, training=training)
        h = self.head_dense3(h)
        h = self.head_drop3(h, training=training)
        rul = self.head_output(h + self.head_skip_dense(pooled))
        koopman_out["spectral_features"] = spec
        return rul, koopman_out

    def call(self, inputs, training=None):
        if self.condition_aware:
            return self._condition_aware_call(inputs, training)
        if self.koopman_kind == "none":
            return self._call_without_koopman(inputs, training)
        if self.head == "kepin":
            return super().call(inputs, training=training)
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        z_seq = self.encode(x, training=training)
        koopman_out = self.koopman(z_seq, training=training)
        z_T = z_seq[:, -1]
        health = self.koopman.health_rollout(z_T, TTF_HORIZON, self.health_w, self.health_b)
        if self.head == "wear":
            v = tf.nn.softplus(tf.tensordot(tf.cast(z_T, tf.float32), self.wear_w, 1) + self.wear_b)[:, None] + 1e-3
            k = tf.range(TTF_HORIZON + 1, dtype=tf.float32)[None, :]
            health = health + v * k                                  # wear accumulator mode
            logits = -(health[:, 1:] / v) / WEAR_TAU                 # crossing measured in flights
            koopman_out["wear_rate"] = v
        else:
            logits = -health[:, 1:]
        log_survive = tf.math.cumsum(tf.math.log_sigmoid(logits), axis=1)          # (B, H)
        rul = tf.reduce_sum(tf.exp(log_survive), axis=1, keepdims=True)
        koopman_out["health_trajectory"] = health
        return rul, koopman_out

    def _call_without_koopman(self, inputs, training):
        """KePINModel.call minus the Koopman layer and spectral features (same head)."""
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        z_seq = tf.cast(self.encode(x, training=training), tf.float32)
        pooled = tf.concat([self.gap(z_seq), self.gmp(z_seq)], axis=-1)
        h = self.head_dense1(pooled)
        h = self.head_bn1(h, training=training)
        h = self.head_drop1(h, training=training)
        h = self.head_dense2(h)
        h = self.head_bn2(h, training=training)
        h = self.head_drop2(h, training=training)
        h = self.head_dense3(h)
        h = self.head_drop3(h, training=training)
        rul = self.head_output(h + self.head_skip_dense(pooled))
        # placeholders so the shared loss/trainer code runs; the Koopman losses are excluded
        zero = tf.zeros((tf.shape(x)[0], 1, 1))
        koopman_out = {"one_step_pred": zero, "one_step_target": zero,
                       "multi_step_pred": zero[..., None], "multi_step_target": zero[..., None],
                       "eigenvalues": tf.zeros((1,), tf.complex64), "final_state": z_seq[:, -1]}
        return rul, koopman_out

    def get_eigenvalues(self):
        if self.koopman_kind == "none":
            return np.zeros(1, np.complex64)
        if self.koopman_kind == "eig":
            return self.koopman.eigenvalues().numpy()     # static part
        return super().get_eigenvalues()

    def get_koopman_matrix(self):
        if self.koopman_kind == "none":
            return None
        if self.koopman_kind == "eig":
            return self.koopman.koopman_matrix().numpy()
        return super().get_koopman_matrix()
