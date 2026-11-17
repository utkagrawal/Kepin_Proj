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

Regime-adaptive Koopman (koopman="eig", koopman_mode="health", cond_norm=True):
  The condition-aware input (cond_norm + in-model EMA) feeds the eig operator in health mode.
  A scalar health h per window sets the decay of every mode; with health_map="ordered",
  lower health can only mean faster decay (healthy: slow forgetting, near failure: fast).
  head_state = H > 0 additionally feeds the head the current state carried H flights ahead
  by K(h), so the regime-dependent shrinking directly changes the prediction.
  How h is tied to degradation (monotone over life, or supervised) is set in regime_trainer.py.
  health_source="signal_a": h is not read from the latent state but from condition-normalised
  Signal A, a fixed formula on the raw window: each sensor is z-scored with its operating
  condition's mean/std, then [mean, variance, skewness, kurtosis] over the window, averaged over
  sensors -> 4 numbers A; h = sigmoid(w . standardise(A) + b). Only w (4) and b are learned,
  so the regime that drives K(h) is fully readable.

Condition-aware wear model (head="wear", koopman="eig", koopman_mode="static"|"health", cond_norm=True):
  With koopman_mode="health" the health forecast uses each engine's own operator (adapt = gate /
  mixture / engine), so static vs adaptive K is compared with everything else identical.
  dynamics (2x2 study: linear/nonlinear x static/dynamic latent dynamics, wear model only):
    "linear"     the operator above (closed form).
    "nl_static"  z~(k+1) = K z~(k) + g(z~(k)), g = bounded MLP (tanh, zero-initialised output, so
                 training starts from the static model). Same g for every engine.
    "nl_dynamic" z~(k+1) = K(h(z~(k))) z~(k): the health h is re-read from the CURRENT state at every
                 forecast step (state-level health net), and lower health -> faster decay
                 (ordered gap). The operator changes along the forecast as the engine degrades.
  Both are rolled out flight by flight (125 steps); the one-step Koopman loss uses the same step.
  cond_norm -> EMA -> encoder -> static eig Koopman; RUL = flights until the projected health
  H_k = H_koopman(k) + v * k reaches 0 (no shortcut around K). With health_source="signal_a" the
  wear rate is set by condition-normalised Signal A alone:
      v = softplus(SA_LR * (w . standardise(A) + b)),   A = [mean, var, skew, kurt] (4 numbers)
  so "how fast this engine wears" is a 5-parameter formula of its sensors. SA_LR = 10 is a
  learning-rate multiplier for these 5 numbers (with the shared Adam rate they barely move).
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

SA_LR = 10.0               # learning-rate multiplier of the Signal A wear-rate formula
TTF_HORIZON = 125          # = RUL cap of the C-MAPSS labels
WEAR_TAU = 2.0             # width of the soft threshold crossing, in flights


class DynamicKePIN(KePINModel):

    def __init__(self, input_shape_tuple, arch_config=None, n_train=None, n_active_losses=7,
                 koopman="eig", koopman_mode="static", exo_dim=0, head="kepin",
                 cond_norm=False, koop_control=False, cond_stats=None, ema_alpha=None,
                 health_map="free", head_state=0, health_source="latent", gap_init=-4.0,
                 sig_a_stats=None, adapt="gate", dynamics="linear", encoder="deep",
                 hyper_ctx_dim=0, hyper_rank=False, hist_health=False, use_age=False, head_ctx=False,
                 hist_dim=42, **kwargs):
        super().__init__(input_shape_tuple, arch_config=arch_config, n_train=n_train,
                         n_active_losses=n_active_losses, **kwargs)
        self.koopman_kind = koopman
        self.koopman_mode = koopman_mode
        self.head = head
        if head in ("ttf", "wear") and (koopman != "eig" or (koopman_mode != "static" and not
                                                             (head == "wear" and cond_norm))):
            raise ValueError("the time-to-failure head uses the eig operator (adaptive only with "
                             "the condition-aware wear model)")
        if koopman == "eig":
            self.koopman = EigenKoopmanOperator(
                latent_dim=self.arch_config["latent_dim"],
                rollout_steps=self.arch_config["rollout"],
                mode=koopman_mode, exo_dim=0 if cond_norm else exo_dim, health_map=health_map,
                gap_init=gap_init, external_health=health_source == "signal_a" or dynamics == "nl_dynamic",
                adapt=adapt, hyper_rank=hyper_rank,
                hyper_ctx_dim=(1 + int(use_age)) if hist_health else hyper_ctx_dim,
                hyper_use_window=not hist_health,
                s_init="spread" if head in ("ttf", "wear") else "zeros", name="eigen_koopman")
        elif koopman == "none":
            if head != "kepin" or koopman_mode != "static":
                raise ValueError("koopman='none' is the plain KePIN-head ablation")
            self.koopman = None
        elif koopman != "svd":
            raise ValueError(koopman)
        self.cond_norm, self.koop_control = cond_norm, koop_control
        self.condition_aware = cond_norm or koop_control
        self.regime = koopman == "eig" and koopman_mode == "health" and cond_norm
        # flights ahead at which the head reads the state carried by this engine's operator:
        # an int (one horizon, 0 = none) or a list (forecast head, e.g. [1, 5, 10, 25, 50])
        self.head_states = [int(k) for k in head_state] if isinstance(head_state, (list, tuple)) \
            else ([int(head_state)] if int(head_state) else [])
        self.head_state = len(self.head_states)
        if health_source not in ("latent", "signal_a"):
            raise ValueError(health_source)
        self.sig_a = health_source == "signal_a"
        self.cn_wear = (head == "wear" and koopman == "eig" and koopman_mode in ("static", "health")
                        and cond_norm and not koop_control)
        # encoder="linear": the latent is a plain linear map of the condition-normalised, smoothed
        # sensors (no CNN/LSTM/attention, no normalisation layers), so it cannot bend the
        # degradation curve (the sensors accelerate ~5x near failure; RUL-trained deep latents ~1.8x)
        if encoder not in ("deep", "linear"):
            raise ValueError(encoder)
        self.encoder_kind = encoder
        if encoder == "linear":
            if not self.cn_wear:
                raise ValueError("encoder='linear' is for the condition-aware wear model")
            self.lin_enc = keras.layers.Dense(self.arch_config["latent_dim"], name="linear_encoder")
            self.lin_enc.build((None, input_shape_tuple[0], input_shape_tuple[1]))
        # history-health bottleneck (original KePIN): the second input is [42 engine-history
        # features, age/100]; a small net maps the history features to ONE health index h
        # (trained label-free with the mono regime loss). Hypernetwork Koopman: context = h (+ age)
        # only. head_ctx (control): static K, and [h, age] go straight into the KePIN head.
        self.hist_health, self.use_age, self.head_ctx = bool(hist_health), bool(use_age), bool(head_ctx)
        self.hist_dim = int(hist_dim)   # 42 on C-MAPSS (21 sensors x drift, rate); 2 x n_features elsewhere
        if self.hist_health:
            if self.condition_aware or head != "kepin" or koopman != "eig":
                raise ValueError("hist_health: original KePIN pipeline, eig Koopman, KePIN head")
            if head_ctx == (koopman_mode == "hyper"):
                raise ValueError("hist_health: either mode='hyper' (K reads h) or head_ctx with a static K")
            self.hh_hidden = keras.layers.Dense(32, activation="relu", name="hist_health_hidden")
            self.hh_out = keras.layers.Dense(1, activation="sigmoid", name="hist_health_out")
            self.hh_hidden.build((None, self.hist_dim))
            self.hh_out.build((None, 32))
        if dynamics not in ("linear", "nl_static", "nl_dynamic"):
            raise ValueError(dynamics)
        self.dynamics = dynamics
        if dynamics != "linear":
            want = "static" if dynamics == "nl_static" else "health"
            if not (self.cn_wear and koopman_mode == want and health_source == "latent" and adapt == "gate"
                    and (dynamics == "nl_static" or health_map == "ordered")):
                raise ValueError("nl_static: cond_norm wear model, mode static; nl_dynamic: cond_norm wear "
                                 "model, mode health, health_map ordered")
            d_lat = self.arch_config["latent_dim"]
            if dynamics == "nl_static":
                self.nl_hidden = keras.layers.Dense(64, activation="relu", name="nl_dyn_hidden")
                self.nl_out = keras.layers.Dense(d_lat, kernel_initializer="zeros", bias_initializer="zeros",
                                                 name="nl_dyn_out")
                self.nl_hidden.build((None, d_lat))
                self.nl_out.build((None, 64))
            else:
                self.hs_hidden = keras.layers.Dense(32, activation="relu", name="state_health_hidden")
                self.hs_out = keras.layers.Dense(1, activation="sigmoid", name="state_health_out")
                self.hs_hidden.build((None, d_lat))
                self.hs_out.build((None, 32))
        if self.sig_a and not (self.regime or self.cn_wear):
            raise ValueError("health_source='signal_a' is for the regime-adaptive or cond_norm wear model")
        if self.head_state and not self.regime:
            raise ValueError("head_state is for the regime-adaptive model (eig, health, cond_norm)")
        if self.condition_aware:
            ok_svd = koopman == "svd" and koopman_mode == "static"
            ok_regime = self.regime
            if not (self.cn_wear or (head == "kepin" and (ok_svd or ok_regime))):
                raise ValueError("condition-aware Koopman: svd/static (original KePIN) or "
                                 "eig/health with cond_norm (regime-adaptive), KePIN head; "
                                 "or eig/static wear head with cond_norm")
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
            if self.sig_a:
                # exact per-condition mean/std of the 21 sensors (fixed); constant sensors masked
                mu_a, sd_a = sig_a_stats
                mu_a, sd_a = mu_a[:, 3:], sd_a[:, 3:]
                keep = (sd_a > 1e-6).all(0).astype(np.float32)
                self.sa_mu = tf.constant(mu_a, tf.float32)
                self.sa_inv_sd = tf.constant(np.where(sd_a > 1e-6, 1.0 / np.maximum(sd_a, 1e-6), 0.0), tf.float32)
                self.sa_keep = tf.constant(keep / keep.sum(), tf.float32)        # channel average
                # standardisation of the 4 moments (set by set_signal_a_scale), saved with weights
                self.sa_fmu = self.add_weight(name="sigA_feat_mean", shape=(4,), initializer="zeros",
                                              trainable=False)
                self.sa_fsd = self.add_weight(name="sigA_feat_std", shape=(4,), initializer="ones",
                                              trainable=False)
                self.sa_w = self.add_weight(name="sigA_w", shape=(4,), initializer="zeros")
                # wear model: v = softplus(SA_LR * (w.F + b)); b = -0.248 -> v ~ 0.08 at the start
                self.sa_b = self.add_weight(name="sigA_b", shape=(), initializer=keras.initializers.Constant(
                    -2.48 / SA_LR if self.cn_wear else 0.0))
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
        if head == "wear" and not self.sig_a:
            # wear rate v = softplus(w_v . z_T + b_v); init v ~ 0.08 -> first crossings ~60 flights
            self.wear_w = self.add_weight(name="wear_rate_w", shape=(d,),
                                          initializer=keras.initializers.RandomNormal(stddev=0.01))
            self.wear_b = self.add_weight(name="wear_rate_b", shape=(),
                                          initializer=keras.initializers.Constant(-2.48))

    # ---- encoder freezing (self-supervised pre-training, frozen-encoder study) -------------
    def encoder_layers(self):
        """Every layer used by encode(), in a fixed order."""
        L = [self.input_proj, self.input_bn]
        for block in self.encoder_blocks:
            L += [block[k] for k in ("conv1", "bn1", "conv2", "bn2", "se_gap", "se_dense1", "se_dense2",
                                     "se_reshape", "se_mul", "skip_proj", "skip_bn") if block.get(k) is not None]
        L += [self.bilstm, self.lstm_ln, self.post_lstm_proj, self.post_lstm_bn, self.mha, self.mha_dropout,
              self.mha_ln, self.ff_dense1, self.ff_dense2, self.ff_dropout, self.ff_ln, self.latent_proj,
              self.latent_bn]
        return L

    def encoder_weights(self):
        """All encoder weights (incl. BatchNorm statistics) + the condition-normalisation step."""
        W = [w for l in self.encoder_layers() for w in l.get_weights()]
        return W + [self.cn_gamma.numpy(), self.cn_beta.numpy()]

    def load_frozen_encoder(self, path):
        """Load encoder (+ cond_norm) weights saved by pretrain_ssl.py and freeze them."""
        f = np.load(path)
        W = [f[f"w{i}"] for i in range(int(f["n"]))]
        i = 0
        for l in self.encoder_layers():
            n = len(l.get_weights())
            if n:
                for a, b in zip(l.get_weights(), W[i:i + n]):
                    if a.shape != b.shape:
                        raise ValueError(f"encoder weight shape mismatch in {l.name}: {a.shape} vs {b.shape}")
                l.set_weights(W[i:i + n])
            i += n
        self.cn_gamma.assign(W[i]); self.cn_beta.assign(W[i + 1])
        for l in self.encoder_layers():
            l.trainable = False          # BatchNorm then also runs in inference mode
        self.cn_gamma.trainable = False
        self.cn_beta.trainable = False

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

    def signal_a(self, inputs):
        """Condition-normalised Signal A of each raw window: (B, 4) = [mean, var, skew, kurt]."""
        x, u_flat = inputs
        x = tf.cast(x, tf.float32)[..., 3:]                                     # 21 sensors
        u = tf.reshape(tf.cast(u_flat, tf.float32), [-1, x.shape[1], self.n_cond])
        xn = (x - tf.einsum("btc,cf->btf", u, self.sa_mu)) * tf.einsum("btc,cf->btf", u, self.sa_inv_sd)
        m = tf.reduce_mean(xn, 1)
        c = xn - m[:, None]
        var = tf.reduce_mean(c ** 2, 1)
        skew = tf.reduce_mean(c ** 3, 1) / (var ** 1.5 + 1e-4)                  # -> 0 if flat
        kurt = (tf.reduce_mean(c ** 4, 1) - 3 * var ** 2) / (var ** 2 + 1e-4)   # excess; 0 if flat
        return tf.stack([tf.reduce_sum(v * self.sa_keep, -1) for v in (m, var, skew, kurt)], -1)

    def set_signal_a_scale(self, inputs, batch=4096):
        X, U = inputs
        A = np.concatenate([self.signal_a((X[i:i + batch], U[i:i + batch])).numpy()
                            for i in range(0, len(X), batch)])
        self.sa_fmu.assign(A.mean(0))
        self.sa_fsd.assign(np.maximum(A.std(0), 1e-6))
        return A

    def signal_a_health(self, inputs):
        F = tf.clip_by_value((self.signal_a(inputs) - self.sa_fmu) / self.sa_fsd, -5.0, 5.0)
        return tf.sigmoid(tf.reduce_sum(F * self.sa_w, -1, keepdims=True) + self.sa_b)

    def _condition_aware_latent(self, inputs, training):
        """cond_norm -> in-model EMA -> encoder. Returns the latent sequence and the one-hots."""
        x, u_flat = inputs
        x = tf.cast(x, tf.float32)
        u = tf.reshape(tf.cast(u_flat, tf.float32), [-1, x.shape[1], self.n_cond])   # (B, T, C)
        if self.cond_norm:
            x = x * tf.einsum("btc,cf->btf", u, self.cn_gamma) + tf.einsum("btc,cf->btf", u, self.cn_beta)
        x = tf.einsum("ts,bsf->btf", self.ema_matrix, x)          # KePIN's EMA, after the condition step
        z_seq = tf.cast(self.lin_enc(x) if self.encoder_kind == "linear" else self.encode(x, training=training),
                        tf.float32)
        if self.koop_control:   # control term: remove the condition's offset, z = s + C u
            z_seq = z_seq - tf.einsum("btc,cd->btd", u, self.control_C)
        return z_seq, u

    def history_health(self, c):
        """Health index h (batch, 1) from the 42 engine-history features (first 42 context columns)."""
        return self.hh_out(self.hh_hidden(tf.cast(c, tf.float32)[:, :self.hist_dim]))

    def regime_health(self, inputs, training=None):
        """Health index h (batch, 1) of the regime-adaptive model, without the RUL head."""
        if self.hist_health:
            return self.history_health(inputs[1])
        if self.sig_a:
            return self.signal_a_health(inputs)
        if self.dynamics == "nl_dynamic":          # state-level health of the window's last state
            z_seq, _ = self._condition_aware_latent(inputs, training)
            return self.state_health(tf.matmul(z_seq[:, -1], self.koopman.basis()))
        z_seq, _ = self._condition_aware_latent(inputs, training)
        return self.koopman.health(z_seq)

    def _condition_aware_call(self, inputs, training):
        z_seq, u = self._condition_aware_latent(inputs, training)      # condition-free if ctrl
        h_ext = self.signal_a_health(inputs) if self.sig_a else None
        koopman_out = (self.koopman(z_seq, signal=None, h=h_ext, training=training) if self.koopman_kind == "eig"
                       else self.koopman(z_seq, training=training))       # svd layer has no h
        pooled = tf.concat([self.gap(z_seq), self.gmp(z_seq)], axis=-1)
        spec = tf.cast(extract_spectral_features(koopman_out["eigenvalues"], top_k=self.spectral_k),
                       tf.float32)
        if len(spec.shape) == 1:                                   # fixed operator: same for all
            spec = tf.tile(tf.expand_dims(spec, 0), [tf.shape(z_seq)[0], 1])
        head_in = [pooled, spec]
        for k in self.head_states:                                 # state carried ahead by K(engine)
            head_in.append(self.koopman.propagate(z_seq[:, -1], koopman_out["adapt_log_rho"],
                                                  koopman_out["adapt_theta"], k))
        h = self.head_dense1(tf.concat(head_in, axis=-1))
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

    def wear_rate(self, inputs, z_T):
        """Per-engine wear rate v (B, 1): from the latent state, or from Signal A."""
        if self.sig_a:
            F = tf.clip_by_value((self.signal_a(inputs) - self.sa_fmu) / self.sa_fsd, -5.0, 5.0)
            return tf.nn.softplus(SA_LR * (tf.reduce_sum(F * self.sa_w, -1, keepdims=True) + self.sa_b)) + 1e-3
        return tf.nn.softplus(tf.tensordot(tf.cast(z_T, tf.float32), self.wear_w, 1) + self.wear_b)[:, None] + 1e-3

    # ---- nonlinear latent dynamics (2x2 study) -------------------------------------------
    def nl_residual(self, zt):
        """Bounded nonlinear correction g(z~) in modal coordinates (any leading shape)."""
        return tf.tanh(self.nl_out(self.nl_hidden(zt)))

    def state_health(self, zt):
        """Health read from a single (modal) state, (..., 1) in (0, 1)."""
        return self.hs_out(self.hs_hidden(zt))

    def nl_step(self, zt):
        """One flight of the nonlinear dynamics in modal coordinates. Returns (next state, h or None)."""
        K, m = self.koopman, self.koopman.n_modes
        if self.dynamics == "nl_static":
            a, b = K._apply(zt[..., :m], zt[..., m:], tf.math.log_sigmoid(K.s), K.theta)
            return tf.concat([a, b], -1) + self.nl_residual(zt), None
        h = self.state_health(zt)
        log_rho = tf.math.log_sigmoid(K.s + K.health_shift(h))
        a, b = K._apply(zt[..., :m], zt[..., m:], log_rho, K.theta)
        return tf.concat([a, b], -1), h

    def _nl_health_rollout(self, z_T):
        """Health forecast H_0..H_125 by stepping the nonlinear dynamics flight by flight."""
        zt = tf.matmul(tf.cast(z_T, tf.float32), self.koopman.basis())
        Hs, hs = [tf.tensordot(zt, self.health_w, 1) + self.health_b], []
        for _ in range(TTF_HORIZON):
            zt, h = self.nl_step(zt)
            Hs.append(tf.tensordot(zt, self.health_w, 1) + self.health_b)
            if h is not None:
                hs.append(h[:, 0])
        return tf.stack(Hs, 1), (tf.stack(hs, 1) if hs else None)

    def _wear_head(self, koopman_out, z_T, v, log_rho=None, theta=None):
        if self.dynamics == "linear":
            health = self.koopman.health_rollout(z_T, TTF_HORIZON, self.health_w, self.health_b,
                                                 log_rho, theta)
        else:
            health, h_path = self._nl_health_rollout(z_T)
            if h_path is not None:
                koopman_out["forecast_health_index"] = h_path       # h along the forecast (B, 125)
        k = tf.range(TTF_HORIZON + 1, dtype=tf.float32)[None, :]
        health = health + v * k                                      # wear accumulator mode
        logits = -(health[:, 1:] / v) / WEAR_TAU                     # crossing measured in flights
        log_survive = tf.math.cumsum(tf.math.log_sigmoid(logits), axis=1)
        rul = tf.reduce_sum(tf.exp(log_survive), axis=1, keepdims=True)
        koopman_out["wear_rate"] = v
        koopman_out["health_trajectory"] = health
        return rul, koopman_out

    def call(self, inputs, training=None):
        if self.cn_wear:     # condition-aware wear model: the RUL comes only through K
            z_seq, _ = self._condition_aware_latent(inputs, training)
            h_ext = None
            if not self.koopman.built:                 # basis() is needed before the first call
                self.koopman.build(z_seq.shape)
            if self.dynamics == "nl_dynamic":
                zt_seq = tf.matmul(z_seq, self.koopman.basis())
                h_ext = self.state_health(zt_seq[:, -1])
            koopman_out = self.koopman(z_seq, h=h_ext, training=training)
            if self.dynamics != "linear":   # one-step Koopman loss with the SAME nonlinear step
                zt_seq = tf.matmul(z_seq, self.koopman.basis())
                koopman_out["one_step_pred"], _ = self.nl_step(zt_seq[:, :-1])
            z_T = z_seq[:, -1]
            adaptive = self.koopman_mode != "static" and self.dynamics == "linear"
            return self._wear_head(koopman_out, z_T, self.wear_rate(inputs, z_T),
                                   koopman_out["adapt_log_rho"] if adaptive else None,
                                   koopman_out["adapt_theta"] if adaptive else None)
        if self.condition_aware:
            return self._condition_aware_call(inputs, training)
        if self.koopman_kind == "none":
            return self._call_without_koopman(inputs, training)
        if self.hist_health:
            x, c = inputs
            h = self.history_health(c)
            age = tf.cast(c, tf.float32)[:, self.hist_dim:self.hist_dim + 1]
            if self.head_ctx:                               # control: static K, [h, age] -> head
                self._head_extra = tf.concat([h, age], -1)
                rul, ko = super().call(x, training=training)
            else:                                           # K is generated from h (+ age) only
                self._head_extra = None
                sig = tf.concat([h, age], -1) if self.use_age else h
                rul, ko = super().call((x, sig), training=training)
            ko["health"] = h
            return rul, ko
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
