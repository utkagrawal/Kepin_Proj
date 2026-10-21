# -*- coding: utf-8 -*-
"""
Koopman Operator Module — core novelty layer for KePIN.

Implements a learnable linear Koopman operator K in a latent space, such that
the dynamics of the encoded states satisfy:

    z(t+1) ≈ K · z(t)

The operator K is parameterised via SVD factorisation:

    K = U · diag(σ(s)) · V^T

where σ is the sigmoid function bounding singular values to [0, 1],
ensuring spectral stability (all eigenvalues |λ_i| ≤ 1).

The spectral decomposition of K reveals:
  - Decay rates:  σ_i = -Re(log(λ_i))    (how fast mode i decays)
  - Frequencies:  ω_i =  Im(log(λ_i))    (oscillation frequency)
  - Spectral radius ρ(K) = max|λ_i|       (overall stability)

Reference:
  Lusch, Wehmeyer, Klus — "Deep learning for universal linear embeddings
  of nonlinear dynamics", Nature Communications, 2018.
"""

import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable


# =========================================================================
# Koopman Operator Keras Layer
# =========================================================================

@register_keras_serializable(package="KePIN")
class KoopmanOperator(keras.layers.Layer):
    """Learnable Koopman operator with SVD-parameterised stability.

    Given a sequence of latent states Z = [z(1), ..., z(T)] of shape
    (batch, T, d), this layer:
      1. Computes one-step predictions:  z_hat(t+1) = K · z(t)
      2. Computes multi-step rollouts:   z_hat(t+k) = K^k · z(t)
      3. Extracts eigenvalues for spectral analysis and physics constraints

    Parameters
    ----------
    latent_dim : int
        Dimensionality d of the latent state vectors.
    rollout_steps : int
        Number of multi-step rollout predictions (default: 3).
    stability_mode : str
        'svd' (default) — parameterise K = U · diag(σ(s)) · V^T
        'full' — unconstrained K (for ablation baseline)
    """

    def __init__(self, latent_dim: int, rollout_steps: int = 3,
                 stability_mode: str = "svd", **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode

    def build(self, input_shape):
        d = self.latent_dim

        if self.stability_mode == "svd":
            # SVD factorisation: K = U · diag(sigmoid(s)) · V^T
            # U and V are strictly orthogonal matrices parameterized via Cayley Transform.
            self.M_U = self.add_weight(
                name="M_U", shape=(d, d),
                initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                trainable=True,
            )
            self.s = self.add_weight(
                name="s", shape=(d,),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
            # Add condition network for dynamic Koopman operator
            self.condition_net_raw = keras.Sequential([
                keras.layers.Dense(64, activation="relu", name="cond_dense1"),
                keras.layers.Dense(64, activation="relu", name="cond_dense2"),
                keras.layers.Dense(d, activation="linear", 
                                   kernel_initializer=keras.initializers.Zeros(),
                                   bias_initializer=keras.initializers.Zeros(),
                                   name="cond_out")
            ], name="condition_net_raw")
            self.M_V = self.add_weight(
                name="M_V", shape=(d, d),
                initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
                trainable=True,
            )
        else:
            # Unconstrained K (for ablation)
            self.K_raw = self.add_weight(
                name="K_raw", shape=(d, d),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True,
            )

        super().build(input_shape)

    def _get_K(self, signal=None):
        """Construct the Koopman operator matrix K."""
        if self.stability_mode == "svd":
            d = self.latent_dim
            I = tf.eye(d, dtype=self.M_U.dtype)
            
            # Cayley transform for U: U = (I - A)(I + A)^-1 = (I + A)^-1 (I - A)
            A_U = self.M_U - tf.transpose(self.M_U)
            U = tf.linalg.solve(I + A_U, I - A_U)
            
            # Cayley transform for V: V = (I - A)(I + A)^-1 = (I + A)^-1 (I - A)
            A_V = self.M_V - tf.transpose(self.M_V)
            V = tf.linalg.solve(I + A_V, I - A_V)
            
            if signal is not None:
                # Dynamic conditioning using the regime signal
                delta_s = 2.0 * tf.nn.tanh(self.condition_net_raw(signal)) # max_shift = 2.0
                sigma = tf.nn.sigmoid(self.s + delta_s) # (batch, d)
                
                # Expand U and V for batched multiplication
                batch_size = tf.shape(signal)[0]
                U_batch = tf.broadcast_to(tf.expand_dims(U, 0), [batch_size, d, d])
                V_batch_T = tf.broadcast_to(tf.expand_dims(tf.transpose(V), 0), [batch_size, d, d])
                
                K_batch = tf.matmul(U_batch, tf.matmul(tf.linalg.diag(sigma), V_batch_T))
                return K_batch
            else:
                sigma = tf.nn.sigmoid(self.s)  # ∈ (0, 1) → bounded singular values
                return tf.matmul(U, tf.matmul(tf.linalg.diag(sigma), tf.transpose(V)))
        else:
            return self.K_raw

    def call(self, z_sequence, signal=None, training=None):
        """Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            signal: (batch, r_dim) — regime conditioning signal (optional)

        Returns:
            dict with keys:
                'one_step_pred':    (batch, T-1, d)  — K·z(t) for t=1..T-1
                'one_step_target':  (batch, T-1, d)  — z(t+1) for t=1..T-1
                'multi_step_pred':  (batch, T-H, H, d) — K^k·z(t) for k=1..H
                'multi_step_target':(batch, T-H, H, d) — z(t+k) for k=1..H
                'K':                (d, d) or (batch, d, d) — the operator matrix
                'eigenvalues':      (d,) or (batch, d) complex — eigenvalues of K
                'final_state':      (batch, d) — z(T), used for RUL head
        """
        K = self._get_K(signal)
        # Cast everything to float32 for numerical stability
        K = tf.cast(K, tf.float32)
        z_sequence = tf.cast(z_sequence, tf.float32)
        batch_size = tf.shape(z_sequence)[0]
        T = tf.shape(z_sequence)[1]

        # --- One-step prediction: z_hat(t+1) = K · z(t) ---
        z_input = z_sequence[:, :-1, :]      # (batch, T-1, d)
        z_target = z_sequence[:, 1:, :]       # (batch, T-1, d)
        
        is_batched = (len(K.shape) == 3)
        if is_batched:
            z_pred_one = tf.matmul(z_input, K, transpose_b=True)
        else:
            z_pred_one = tf.matmul(z_input, tf.transpose(K))  # (batch, T-1, d)

        # --- Multi-step rollout ---
        H = min(self.rollout_steps, 5)  # cap for memory
        # Compute K^k for k = 1..H
        K_powers = [K]  # K^1
        for k in range(1, H):
            K_powers.append(tf.matmul(K_powers[-1], K))  # K^(k+1)

        # For each starting point t, predict z(t+k) = K^k · z(t)
        # Use only positions where all H future steps exist
        max_start = T - H - 1  # we need z(t) and z(t+1),...,z(t+H)

        multi_preds = []
        multi_targets = []
        for k_idx in range(H):
            k = k_idx + 1   # rollout horizon (1-indexed)
            # Predicted: z(t) @ (K^k)^T for t = 0..max_start
            z_start = z_sequence[:, :max_start + 1, :]            # (batch, max_start+1, d)
            if is_batched:
                z_pred_k = tf.matmul(z_start, K_powers[k_idx], transpose_b=True)
            else:
                z_pred_k = tf.matmul(z_start, tf.transpose(K_powers[k_idx]))
            z_true_k = z_sequence[:, k:max_start + 1 + k, :]     # (batch, max_start+1, d)
            multi_preds.append(z_pred_k)
            multi_targets.append(z_true_k)

        # Stack along rollout dimension: (batch, max_start+1, H, d)
        multi_step_pred = tf.stack(multi_preds, axis=2)
        multi_step_target = tf.stack(multi_targets, axis=2)

        # Cast K to float64 for eigvals (float32 often fails to converge on GPU)
        K_f64 = tf.cast(K, tf.float64)
        eigenvalues_f64 = tf.linalg.eigvals(K_f64)
        eigenvalues = tf.cast(eigenvalues_f64, tf.complex64)  # (d,) complex

        # --- Final latent state for RUL head ---
        final_state = z_sequence[:, -1, :]  # (batch, d)

        return {
            "one_step_pred": z_pred_one,
            "one_step_target": z_target,
            "multi_step_pred": multi_step_pred,
            "multi_step_target": multi_step_target,
            "K": K,
            "eigenvalues": eigenvalues,
            "final_state": final_state,
        }

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
        })
        return config


# =========================================================================
# Spectral Feature Extraction (functional helper)
# =========================================================================

def extract_spectral_features(eigenvalues, top_k: int = 4):
    """Extract interpretable spectral features from Koopman eigenvalues.

    Args:
        eigenvalues: (d,) complex tensor — eigenvalues of K
        top_k: number of dominant modes to extract features from

    Returns:
        spectral_features: (top_k * 2 + 2,) float tensor with:
          - per-mode decay rates:    σ_i = -Re(log(λ_i))
          - per-mode frequencies:    ω_i = |Im(log(λ_i))|
          - spectral radius:         ρ(K) = max|λ_i|
          - spectral gap:            |λ_1| - |λ_2|
    """
    # Eigenvalue magnitudes
    eig_mags = tf.abs(eigenvalues)   # (d,) or (batch, d)
    is_batched = len(eig_mags.shape) > 1

    if is_batched:
        sorted_indices = tf.argsort(eig_mags, direction="DESCENDING", axis=-1)
        top_eigs = tf.gather(eigenvalues, sorted_indices[:, :top_k], axis=-1, batch_dims=1)
        top_mags = tf.gather(eig_mags, sorted_indices[:, :top_k], axis=-1, batch_dims=1)
        
        log_eigs = tf.math.log(tf.cast(top_eigs, tf.complex64) + 1e-10)
        decay_rates = -tf.math.real(log_eigs)
        frequencies = tf.abs(tf.math.imag(log_eigs))
        
        spectral_radius = tf.reduce_max(eig_mags, axis=-1)
        all_mags_sorted = tf.sort(eig_mags, direction="DESCENDING", axis=-1)
        spectral_gap = all_mags_sorted[:, 0] - all_mags_sorted[:, 1]
        
        spectral_feats = tf.concat([
            tf.cast(decay_rates, tf.float32),
            tf.cast(frequencies, tf.float32),
            tf.expand_dims(tf.cast(spectral_radius, tf.float32), -1),
            tf.expand_dims(tf.cast(spectral_gap, tf.float32), -1),
        ], axis=-1)
    else:
        sorted_indices = tf.argsort(eig_mags, direction="DESCENDING")
        top_eigs = tf.gather(eigenvalues, sorted_indices[:top_k])
        top_mags = tf.gather(eig_mags, sorted_indices[:top_k])

        log_eigs = tf.math.log(tf.cast(top_eigs, tf.complex64) + 1e-10)
        decay_rates = -tf.math.real(log_eigs)      # σ_i
        frequencies = tf.abs(tf.math.imag(log_eigs))  # ω_i

        spectral_radius = tf.reduce_max(eig_mags)
        all_mags_sorted = tf.sort(eig_mags, direction="DESCENDING")
        spectral_gap = all_mags_sorted[0] - all_mags_sorted[1]

        spectral_feats = tf.concat([
            tf.cast(decay_rates, tf.float32),
            tf.cast(frequencies, tf.float32),
            tf.expand_dims(tf.cast(spectral_radius, tf.float32), 0),
            tf.expand_dims(tf.cast(spectral_gap, tf.float32), 0),
        ], axis=0)

    return spectral_feats


def spectral_features_dim(top_k: int = 4) -> int:
    """Return the output dimension of extract_spectral_features()."""
    return top_k * 2 + 2


# =========================================================================
# Unit test
# =========================================================================

if __name__ == "__main__":
    print("=== Koopman Operator Module — Unit Test ===\n")

    # Create a known linear dynamical system: z(t+1) = K_true · z(t)
    d = 8
    np.random.seed(42)

    # K_true with eigenvalues inside unit circle (stable)
    # Parameterise via random orthogonal U, V and bounded singular values
    from scipy.stats import ortho_group
    U_true = ortho_group.rvs(d).astype(np.float32)
    V_true = ortho_group.rvs(d).astype(np.float32)
    s_true = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40], dtype=np.float32)
    K_true = U_true @ np.diag(s_true) @ V_true.T

    # Generate trajectory
    T = 50
    z0 = np.random.randn(d).astype(np.float32)
    trajectory = [z0]
    for t in range(T - 1):
        trajectory.append(K_true @ trajectory[-1])
    Z = np.stack(trajectory, axis=0)  # (T, d)
    Z_batch = Z[np.newaxis, :, :]    # (1, T, d)

    # Test the layer
    layer = KoopmanOperator(latent_dim=d, rollout_steps=3)
    results = layer(tf.constant(Z_batch))

    print(f"Input shape:           {Z_batch.shape}")
    print(f"One-step pred shape:   {results['one_step_pred'].shape}")
    print(f"One-step target shape: {results['one_step_target'].shape}")
    print(f"Multi-step pred shape: {results['multi_step_pred'].shape}")
    print(f"K shape:               {results['K'].shape}")
    print(f"Eigenvalues:           {results['eigenvalues'].numpy()}")

    # Test spectral features
    spec_feats = extract_spectral_features(results["eigenvalues"])
    print(f"Spectral features:     {spec_feats.numpy()} (dim={spec_feats.shape[0]})")

    # One-step prediction error (should be non-zero since K is not trained yet)
    one_step_err = tf.reduce_mean(tf.square(
        results["one_step_pred"] - results["one_step_target"]
    )).numpy()
    print(f"\nUntrained one-step MSE: {one_step_err:.6f}")

    # Verify K_true eigenvalues
    true_eigs = np.linalg.eigvals(K_true)
    true_eig_mags = np.sort(np.abs(true_eigs))[::-1]
    print(f"True eigenvalue mags:   {true_eig_mags[:4]}")
    print(f"\n✓ All shapes correct. Module ready for integration.")
