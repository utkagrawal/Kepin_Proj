# -*- coding: utf-8 -*-
"""
Koopman Operator Module — core novelty layer for KePIN.

Implements a learnable linear Koopman operator K in a latent space:

    z(t+1) ≈ K · z(t)

K is parameterised via SVD factorisation:

    K = U · diag(σ(s)) · V^T

where σ is the sigmoid function bounding singular values to [0, 1],
ensuring spectral stability (all eigenvalues |λ_i| ≤ 1).
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable


@register_keras_serializable(package="KePIN")
class KoopmanOperator(keras.layers.Layer):
    """Learnable Koopman operator with SVD-parameterised stability.

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
            self.U = self.add_weight(name="U", shape=(d, d),
                                     initializer=keras.initializers.Orthogonal(),
                                     trainable=True)
            if self.condition_dim > 0:
                self.condition_net = keras.Sequential([
                    keras.layers.Dense(max(32, d // 2), activation="relu", kernel_initializer="he_normal"),
                    keras.layers.Dense(d, kernel_initializer="zeros")
                ], name="koopman_condition_net")
            else:
                self.s = self.add_weight(name="s", shape=(d,),
                                         initializer=keras.initializers.Zeros(),
                                         trainable=True)
            self.V = self.add_weight(name="V", shape=(d, d),
                                     initializer=keras.initializers.Orthogonal(),
                                     trainable=True)
        else:
            self.K_raw = self.add_weight(name="K_raw", shape=(d, d),
                                         initializer=keras.initializers.GlorotUniform(),
                                         trainable=True)
        super().build(input_shape)

    def _get_K(self, condition=None):
        """Construct the Koopman operator matrix K."""
        if self.stability_mode == "svd":
            if self.condition_dim > 0:
                if condition is None:
                    condition = tf.zeros((1, self.condition_dim))
                s_batch = self.condition_net(condition)
                sigma = tf.cast(tf.nn.sigmoid(s_batch), tf.float32)
                U_batch = tf.cast(tf.expand_dims(self.U, 0), tf.float32)
                V_batch = tf.cast(tf.expand_dims(self.V, 0), tf.float32)
                diag_sigma = tf.linalg.diag(sigma)
                return tf.matmul(U_batch, tf.matmul(diag_sigma, tf.transpose(V_batch, perm=[0, 2, 1])))
            else:
                sigma = tf.cast(tf.nn.sigmoid(self.s), tf.float32)
                U_f32 = tf.cast(self.U, tf.float32)
                V_f32 = tf.cast(self.V, tf.float32)
                return tf.matmul(U_f32, tf.matmul(tf.linalg.diag(sigma),
                                                    tf.transpose(V_f32)))
        return tf.cast(self.K_raw, tf.float32)

    def call(self, z_sequence, condition=None, training=None):
        """Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            condition: (batch, condition_dim) — optional condition vector

        Returns:
            dict with keys: one_step_pred, one_step_target, multi_step_pred,
            multi_step_target, K, eigenvalues, final_state
        """
        K = tf.cast(self._get_K(condition=condition), tf.float32)
        z_sequence = tf.cast(z_sequence, tf.float32)
        T = tf.shape(z_sequence)[1]

        def _transpose_K(mat):
            return tf.transpose(mat, perm=[0, 2, 1]) if len(mat.shape) == 3 else tf.transpose(mat)

        # One-step prediction: z_hat(t+1) = K · z(t)
        z_input = z_sequence[:, :-1, :]
        z_target = z_sequence[:, 1:, :]
        z_pred_one = tf.matmul(z_input, _transpose_K(K))

        # Multi-step rollout
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

        # Eigenvalues for spectral analysis
        eigenvalues = tf.linalg.eigvals(tf.cast(K, tf.float32))

        return {
            "one_step_pred": z_pred_one,
            "one_step_target": z_target,
            "multi_step_pred": multi_step_pred,
            "multi_step_target": multi_step_target,
            "K": K,
            "eigenvalues": eigenvalues,
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


def extract_spectral_features(eigenvalues, top_k: int = 4):
    """Extract interpretable spectral features from Koopman eigenvalues.

    Supports both unbatched (d,) and batched (batch, d) eigenvalues.
    Returns a float tensor of shape (top_k * 2 + 2,) or (batch, top_k * 2 + 2).
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
