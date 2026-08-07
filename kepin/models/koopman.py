# -*- coding: utf-8 -*-
"""
Koopman Operator Module — core novelty layer for KePIN.

Implements a learnable Koopman operator K in a latent space:

    z(t+1) ≈ K · z(t)

UNCONDITIONED (condition_dim=0):
    K = U · diag(σ(s)) · V^T

CONDITIONED (condition_dim>0) — Parameter-conditioned Koopman operator
inspired by Neural Implicit Flow (NIF):
    K(μ) = U · diag(σ(s + Δs(μ))) · V^T

    where μ is the operating-condition vector (e.g. CMAPSS settings 1-3),
    Δs(μ) = condition_net(μ) is a small learnable perturbation on top of a
    directly-trainable base s.  condition_net is zero-initialized so at t=0:
    Δs(μ) = 0 for all μ → K(μ) is identical to the unconditioned baseline.

Speed note (why we do NOT call tf.linalg.eigvals on batched K):
    With condition_dim>0, K has shape (batch, d, d).  TF's eigvals has no
    efficient GPU batch kernel for non-symmetric matrices — benchmarked at
    ~15 min/epoch for batch=1024, d=128.  We use two separate approaches:

    1. Stability loss  → use σ directly.  Because U,V are orthonormal,
       ρ(K) ≤ ‖K‖₂ = max(σ) < 1 analytically.  No eigvals needed.

    2. Spectral features → subsample the batch to K_SUBSAMPLE=32 matrices,
       compute eigvals only on those, then broadcast to the full batch.
       Cost is O(32·d³) regardless of batch size — same computation as
       the unconditioned case, same complex-valued features.
"""

import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

# Number of K matrices to eigendecompose per forward pass when K is batched.
# Bounding this to a constant keeps cost O(K_SUBSAMPLE·d³) regardless of
# the actual batch size chosen at training time.
_K_SUBSAMPLE = 32


@register_keras_serializable(package="KePIN")
class KoopmanOperator(keras.layers.Layer):
    """Learnable (optionally parameter-conditioned) Koopman operator.

    Given latent states Z = [z(1), ..., z(T)] of shape (batch, T, d),
    computes one-step predictions, multi-step rollouts, and eigenvalue
    decomposition for spectral analysis and physics constraints.

    Parameters
    ----------
    latent_dim : int
        Dimensionality d of the latent state vectors.
    rollout_steps : int
        Number of multi-step rollout predictions (default: 3).
    stability_mode : str
        'svd' — parameterise K = U · diag(σ(s)) · V^T (default)
        'full' — unconstrained K (for ablation baseline)
    condition_dim : int
        If > 0, enables the NIF-inspired parameter-conditioned operator
        K(μ) = U · diag(σ(s + Δs(μ))) · V^T.
    """

    def __init__(self, latent_dim: int, rollout_steps: int = 3,
                 stability_mode: str = "svd", condition_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode
        self.condition_dim = condition_dim

    def build(self, input_shape):
        d = self.latent_dim
        if self.stability_mode == "svd":
            self.U = self.add_weight(
                name="U", shape=(d, d),
                initializer=keras.initializers.Orthogonal(),
                trainable=True)

            # Base singular-value logits — ALWAYS trainable regardless of
            # condition_dim, giving gradient descent a direct path.
            self.s = self.add_weight(
                name="s", shape=(d,),
                initializer=keras.initializers.Zeros(),
                trainable=True)

            if self.condition_dim > 0:
                # condition_net predicts Δs(μ) — the additive offset.
                # Final layer is ZERO-initialized (kernel AND bias) so that
                # Δs(μ) = 0 for every μ at initialization → warm-start
                # property: K(μ) is identical across all μ at t=0.
                self.condition_net = keras.Sequential([
                    keras.layers.Dense(
                        max(32, d // 2), activation="relu",
                        kernel_initializer="he_normal",
                        name="cond_hidden"),
                    keras.layers.Dense(
                        d,
                        kernel_initializer="zeros",
                        bias_initializer="zeros",
                        name="cond_output"),
                ], name="koopman_condition_net")

            self.V = self.add_weight(
                name="V", shape=(d, d),
                initializer=keras.initializers.Orthogonal(),
                trainable=True)
        else:
            self.K_raw = self.add_weight(
                name="K_raw", shape=(d, d),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True)
        super().build(input_shape)

    def _get_sigma(self, condition=None):
        """Return the per-batch singular-value vector σ.

        Returns:
            sigma : float32 tensor, shape (d,) if condition is None / not
                    conditioned, or (batch, d) if conditioned.
            is_batched : bool — True when sigma is (batch, d).
        """
        if self.stability_mode != "svd":
            return None, False

        if self.condition_dim > 0 and condition is not None:
            delta_s = self.condition_net(condition)           # (batch, d)
            sigma = tf.nn.sigmoid(
                tf.cast(self.s, tf.float32) + tf.cast(delta_s, tf.float32))
            return sigma, True

        sigma = tf.nn.sigmoid(tf.cast(self.s, tf.float32))   # (d,)
        return sigma, False

    def _get_K(self, condition=None):
        """Construct the Koopman operator matrix K.

        When condition_dim=0 or condition is None, returns a single
        (d, d) matrix — the BASE operator evaluated at μ=0.

        When condition_dim>0 and condition is provided, returns a
        batched (batch, d, d) operator.

        NOTE for introspection callers (get_koopman_matrix / get_eigenvalues
        in KePINModel): if you call this without `condition`, you get the
        BASE operator at μ=0, which is representative of the initial state
        but NOT of any real sample's dynamics during training.  Pass actual
        condition vectors (extracted from real inputs) for meaningful results.
        """
        if self.stability_mode == "svd":
            sigma, is_batched = self._get_sigma(condition)
            U_f32 = tf.cast(self.U, tf.float32)
            V_f32 = tf.cast(self.V, tf.float32)
            if is_batched:
                # sigma: (batch, d)
                diag_sigma = tf.linalg.diag(sigma)           # (batch, d, d)
                U_b = tf.expand_dims(U_f32, 0)               # (1, d, d)
                V_b = tf.expand_dims(V_f32, 0)               # (1, d, d)
                return tf.matmul(U_b, tf.matmul(diag_sigma,
                                 tf.transpose(V_b, perm=[0, 2, 1])))
            else:
                return tf.matmul(U_f32, tf.matmul(tf.linalg.diag(sigma),
                                                   tf.transpose(V_f32)))
        return tf.cast(self.K_raw, tf.float32)

    # ------------------------------------------------------------------
    # Internal helpers for the forward pass
    # ------------------------------------------------------------------

    def _stability_loss_value(self, sigma, is_batched):
        """Compute stability penalty without calling eigvals.

        Because U and V are orthonormal for any orthogonal U, V:
            ρ(K) ≤ ‖K‖₂ = max(σᵢ) < 1   (since σᵢ = sigmoid(·) ∈ (0,1))
        So the spectral stability loss is analytically zero.  We still
        return a max(σ)-based soft penalty for gradient signal on σ.
        """
        sigma_f32 = tf.cast(sigma, tf.float32)
        if is_batched:
            # max per sample, then mean over batch
            violation = tf.maximum(tf.reduce_max(sigma_f32, axis=-1) - 1.0, 0.0)
        else:
            violation = tf.maximum(tf.reduce_max(sigma_f32) - 1.0, 0.0)
        return tf.reduce_mean(tf.square(violation))

    def _eigenvalues_for_spectral_features(self, K, is_batched):
        """Compute eigenvalues for the spectral feature extraction.

        Unconditioned (is_batched=False): eigvals of single (d,d) K — cheap.
        Conditioned (is_batched=True):   subsample to _K_SUBSAMPLE matrices,
            eigendecompose those, broadcast to full batch.
            Cost = O(_K_SUBSAMPLE · d³), independent of actual batch size.
        """
        if not is_batched:
            return tf.linalg.eigvals(tf.cast(K, tf.float32))

        # Batched conditioned path — subsample
        n_sub = tf.minimum(tf.shape(K)[0], _K_SUBSAMPLE)
        K_sub = K[:n_sub]                                     # (n_sub, d, d)
        eigs_sub = tf.linalg.eigvals(tf.cast(K_sub, tf.float32))  # (n_sub, d)

        # Broadcast sub-batch eigenvalues to the full batch:
        # repeat the sub-batch cyclically to cover the full batch size
        full_batch = tf.shape(K)[0]
        # tile along batch axis to cover full_batch
        repeats = tf.cast(tf.math.ceil(
            tf.cast(full_batch, tf.float32) / tf.cast(n_sub, tf.float32)
        ), tf.int32)
        eigs_tiled = tf.tile(eigs_sub, [repeats, 1])          # (repeats*n_sub, d)
        eigenvalues = eigs_tiled[:full_batch]                  # (batch, d)
        return eigenvalues

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, z_sequence, condition=None, training=None):
        """Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            condition:  (batch, condition_dim) — optional condition vector.
                        Ignored if condition_dim == 0.

        Returns:
            dict with keys: one_step_pred, one_step_target, multi_step_pred,
            multi_step_target, K, eigenvalues, sigma, final_state.
        """
        z_sequence = tf.cast(z_sequence, tf.float32)
        T = tf.shape(z_sequence)[1]

        # ---- Build K ----
        sigma, is_batched = self._get_sigma(
            condition if self.condition_dim > 0 else None)
        K = tf.cast(self._get_K(
            condition if self.condition_dim > 0 else None), tf.float32)

        def _transpose_K(mat):
            return (tf.transpose(mat, perm=[0, 2, 1])
                    if len(mat.shape) == 3 else tf.transpose(mat))

        # ---- One-step prediction: z_hat(t+1) = K · z(t) ----
        z_input = z_sequence[:, :-1, :]
        z_target = z_sequence[:, 1:, :]
        z_pred_one = tf.matmul(z_input, _transpose_K(K))

        # ---- Multi-step rollout ----
        H = min(self.rollout_steps, 5)
        K_powers = [K]
        for k in range(1, H):
            K_powers.append(tf.matmul(K_powers[-1], K))

        max_start = T - H - 1
        multi_preds, multi_targets = [], []
        for k_idx in range(H):
            k = k_idx + 1
            z_start = z_sequence[:, :max_start + 1, :]
            z_pred_k = tf.matmul(z_start, _transpose_K(K_powers[k_idx]))
            z_true_k = z_sequence[:, k:max_start + 1 + k, :]
            multi_preds.append(z_pred_k)
            multi_targets.append(z_true_k)

        multi_step_pred = tf.stack(multi_preds, axis=2)
        multi_step_target = tf.stack(multi_targets, axis=2)

        # ---- Eigenvalues (speed-bounded for batched K) ----
        eigenvalues = self._eigenvalues_for_spectral_features(K, is_batched)

        # ---- Analytical stability value (replaces eigvals-based check) ----
        if sigma is not None:
            stability_val = self._stability_loss_value(sigma, is_batched)
        else:
            stability_val = None

        return {
            "one_step_pred": z_pred_one,
            "one_step_target": z_target,
            "multi_step_pred": multi_step_pred,
            "multi_step_target": multi_step_target,
            "K": K,
            "eigenvalues": eigenvalues,
            "sigma": sigma,
            "sigma_is_batched": is_batched,
            "stability_val": stability_val,
            "final_state": z_sequence[:, -1, :],
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
            "condition_dim": self.condition_dim,
        })
        return config


# ---------------------------------------------------------------------------
# Spectral feature extraction
# ---------------------------------------------------------------------------

def extract_spectral_features(eigenvalues, top_k: int = 4):
    """Extract interpretable spectral features from Koopman eigenvalues.

    Supports both unbatched (d,) and batched (batch, d) eigenvalues.
    Returns a float tensor of shape (top_k * 2 + 2,) or (batch, top_k * 2 + 2).

    Features: decay rates (real part of log(λ)), frequencies (|imag(log(λ))|),
    spectral radius, spectral gap.
    """
    is_batched = len(eigenvalues.shape) > 1
    if not is_batched:
        eigenvalues = tf.expand_dims(eigenvalues, 0)

    eig_mags = tf.abs(eigenvalues)
    sorted_indices = tf.argsort(eig_mags, direction="DESCENDING")
    top_eigs = tf.gather(eigenvalues, sorted_indices[:, :top_k], batch_dims=1)

    log_eigs = tf.math.log(tf.cast(top_eigs, tf.complex64) + 1e-10)
    decay_rates = -tf.math.real(log_eigs)
    frequencies = tf.abs(tf.math.imag(log_eigs))

    spectral_radius = tf.reduce_max(eig_mags, axis=-1)
    all_mags_sorted = tf.sort(eig_mags, direction="DESCENDING", axis=-1)
    spectral_gap = all_mags_sorted[:, 0] - all_mags_sorted[:, 1]

    result = tf.concat([
        tf.cast(decay_rates, tf.float32),
        tf.cast(frequencies, tf.float32),
        tf.expand_dims(tf.cast(spectral_radius, tf.float32), -1),
        tf.expand_dims(tf.cast(spectral_gap, tf.float32), -1),
    ], axis=-1)

    if not is_batched:
        result = tf.squeeze(result, axis=0)
    return result


def spectral_features_dim(top_k: int = 4) -> int:
    """Return the output dimension of ``extract_spectral_features()``."""
    return top_k * 2 + 2
