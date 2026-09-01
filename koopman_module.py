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
from kan_layer import KANLayer


# =========================================================================
# Koopman Operator Keras Layer
# =========================================================================


@register_keras_serializable(package="KePIN")
class RawConditioning(keras.layers.Layer):
    def __init__(self, condition_indices, **kwargs):
        super().__init__(**kwargs)
        self.condition_indices = condition_indices
        
    def get_config(self):
        config = super().get_config()
        config.update({"condition_indices": self.condition_indices})
        return config

    def call(self, inputs):
        if self.condition_indices is None or len(self.condition_indices) == 0:
            return None
        mu_seq = tf.gather(inputs, self.condition_indices, axis=-1)
        mu_seq_f32 = tf.cast(mu_seq, tf.float32)
        return tf.reduce_mean(mu_seq_f32, axis=1)

@register_keras_serializable(package="KePIN")
class RegimeEmbedConditioning(keras.layers.Layer):
    def __init__(self, condition_indices, kmeans_centroids, embed_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.condition_indices = condition_indices
        self.embed_dim = embed_dim
        self.kmeans_centroids = np.array(kmeans_centroids, dtype=np.float32) if kmeans_centroids is not None else np.zeros((6, len(condition_indices)), dtype=np.float32)
        
    def build(self, input_shape):
        self.centroids_var = self.add_weight(
            shape=self.kmeans_centroids.shape,
            initializer=keras.initializers.Constant(self.kmeans_centroids),
            trainable=False,
            name="centroids_var"
        )
        self.embedding = keras.layers.Embedding(self.kmeans_centroids.shape[0], self.embed_dim)
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({"condition_indices": self.condition_indices, "embed_dim": self.embed_dim, "kmeans_centroids": self.kmeans_centroids.tolist()})
        return config

    def call(self, inputs):
        mu_seq = tf.gather(inputs, self.condition_indices, axis=-1)
        mu_seq_f32 = tf.cast(mu_seq, tf.float32)
        mu_raw = tf.reduce_mean(mu_seq_f32, axis=1)
        mu_exp = tf.expand_dims(mu_raw, 1)
        cent_exp = tf.expand_dims(self.centroids_var, 0)
        dists = tf.reduce_sum(tf.square(mu_exp - cent_exp), axis=-1)
        regime_ids = tf.argmin(dists, axis=-1)
        return self.embedding(regime_ids)

@register_keras_serializable(package="KePIN")
class HybridConditioning(keras.layers.Layer):
    def __init__(self, embed_indices, raw_indices, kmeans_centroids, embed_dim=3, **kwargs):
        super().__init__(**kwargs)
        self.embed_indices = embed_indices
        self.raw_indices = raw_indices
        self.embed_dim = embed_dim
        self.kmeans_centroids = np.array(kmeans_centroids, dtype=np.float32) if kmeans_centroids is not None else np.zeros((6, len(embed_indices)), dtype=np.float32)
        
    def build(self, input_shape):
        self.embed_layer = RegimeEmbedConditioning(self.embed_indices, self.kmeans_centroids, self.embed_dim)
        self.raw_layer = RawConditioning(self.raw_indices)
        super().build(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_indices": self.embed_indices, 
            "raw_indices": self.raw_indices, 
            "embed_dim": self.embed_dim, 
            "kmeans_centroids": self.kmeans_centroids.tolist()
        })
        return config
        
    def call(self, inputs):
        embed_out = self.embed_layer(inputs)
        raw_out = self.raw_layer(inputs)
        return tf.concat([embed_out, raw_out], axis=-1)

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
                 stability_mode: str = "svd", condition_dim: int = 0,
                 conditioning_strategy: str = "raw3",
                 condition_indices: list = None,
                 kmeans_centroids = None,
                 condition_net_type: str = "mlp",
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode
        self.condition_dim = condition_dim
        self.conditioning_strategy = conditioning_strategy
        self.condition_indices = condition_indices
        self.kmeans_centroids = kmeans_centroids
        self.condition_net_type = condition_net_type

    def build(self, input_shape):
        d = self.latent_dim

        if self.stability_mode == "svd":
            # SVD factorisation: K = U · diag(sigmoid(s + delta_s(mu))) · V^T
            # Cayley transform parameterisation for exact orthogonality
            self.M_U = self.add_weight(
                name="M_U", shape=(d, d),
                initializer=keras.initializers.Orthogonal(),
                trainable=True,
            )
            # Base singular values (always trainable)
            self.s = self.add_weight(
                name="s", shape=(d,),
                initializer=keras.initializers.Zeros(),
                trainable=True,
            )
            self.M_V = self.add_weight(
                name="M_V", shape=(d, d),
                initializer=keras.initializers.Orthogonal(),
                trainable=True,
            )
            
            # Add parameter-conditioning network if needed
            if self.condition_dim > 0:
                # 1. Setup Strategy Layer
                if self.conditioning_strategy in ["raw3", "raw2_ab", "raw2_ac", "raw2_bc", "sensor_topk"]:
                    self.strategy_layer = RawConditioning(self.condition_indices)
                elif self.conditioning_strategy == "regime_embed":
                    self.strategy_layer = RegimeEmbedConditioning(self.condition_indices, self.kmeans_centroids, embed_dim=self.condition_dim)
                elif self.conditioning_strategy == "hybrid":
                    # hybrid passes embed_indices (first 3) and raw_indices (from condition_indices)
                    embed_indices = list(range(3)) # assuming settings are the first 3
                    self.strategy_layer = HybridConditioning(embed_indices, self.condition_indices, self.kmeans_centroids, embed_dim=3)
                
                # 2. Setup MLP or KAN
                if self.condition_net_type == "kan":
                    self.condition_net = keras.Sequential([
                        KANLayer(max(16, d // 4), grid_size=5, name="kan_hidden1"),
                        KANLayer(d, grid_size=5, name="kan_output")
                    ], name="condition_net_kan")
                else:
                    self.condition_net = keras.Sequential([
                        keras.layers.Dense(64, activation="relu"),
                        keras.layers.Dense(64, activation="relu"),
                        keras.layers.Dense(
                            d, 
                            kernel_initializer="zeros", 
                            bias_initializer="zeros",
                            name="delta_s_out"
                        )
                    ], name="condition_net")
        else:
            # Unconstrained K (for ablation)
            self.K_raw = self.add_weight(
                name="K_raw", shape=(d, d),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True,
            )

        super().build(input_shape)

    def _get_K(self, raw_inputs=None):
        """Construct the Koopman operator matrix K.
        If raw_inputs is provided, we compute mu = strategy(raw_inputs).
        """
        d = self.latent_dim
        I = tf.eye(d, dtype=tf.float32)

        if self.stability_mode == "svd":
            # Exact orthogonal matrices via Cayley transform: U = (I - A)(I + A)^-1, A = M - M^T
            M_U_f32 = tf.cast(self.M_U, tf.float32)
            A_U = M_U_f32 - tf.transpose(M_U_f32)
            U = tf.matmul(I - A_U, tf.linalg.inv(I + A_U))
            
            M_V_f32 = tf.cast(self.M_V, tf.float32)
            A_V = M_V_f32 - tf.transpose(M_V_f32)
            V = tf.matmul(I - A_V, tf.linalg.inv(I + A_V))
            
            # Store U and V for output dictionary
            self.U = U
            self.V = V
            s_val = self.s
            mu = None
            if self.condition_dim > 0 and raw_inputs is not None:
                mu = self.strategy_layer(raw_inputs)
                # Add conditional perturbation: s(mu) = s + delta_s(mu)
                delta_s = self.condition_net(mu) # (batch, d)
                s_val = tf.expand_dims(self.s, 0) + delta_s # (batch, d)
                
            sigma = tf.nn.sigmoid(s_val)  # ∈ (0, 1) → bounded singular values
            
            if self.condition_dim > 0 and mu is not None:
                # Batched K computation
                U_batch = tf.expand_dims(self.U, 0) # (1, d, d)
                V_batch = tf.expand_dims(self.V, 0) # (1, d, d)
                # K = U @ diag(sigma) @ V^T
                sigma_diag = tf.linalg.diag(sigma) # (batch, d, d)
                K = tf.matmul(U_batch, tf.matmul(sigma_diag, V_batch, transpose_b=True))
                return K, sigma
            else:
                return tf.matmul(self.U, tf.matmul(tf.linalg.diag(sigma), tf.transpose(self.V))), sigma
        else:
            return self.K_raw, None

    def call(self, z_sequence, raw_inputs=None, training=None):
        """Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            raw_inputs: (batch, T, n_features) raw inputs to derive condition
        """
        K, sigma = self._get_K(raw_inputs=raw_inputs)
        # Cast everything to float32 for numerical stability
        K = tf.cast(K, tf.float32)
        z_sequence = tf.cast(z_sequence, tf.float32)
        batch_size = tf.shape(z_sequence)[0]
        T = tf.shape(z_sequence)[1]
        
        is_batched_K = (len(K.shape) == 3)

        # --- One-step prediction: z_hat(t+1) = K · z(t) ---
        z_input = z_sequence[:, :-1, :]      # (batch, T-1, d)
        z_target = z_sequence[:, 1:, :]       # (batch, T-1, d)
        
        if is_batched_K:
            # z_input: (batch, T-1, d), K: (batch, d, d)
            # We want z_hat(b, t, i) = sum_j z_input(b, t, j) * K(b, i, j) -> K^T for dot
            z_pred_one = tf.einsum("btj,bji->bti", z_input, tf.transpose(K, perm=[0, 2, 1]))
        else:
            # Vectorised: z_hat = z_input @ K^T  (each row z(t) maps to K·z(t))
            z_pred_one = tf.matmul(z_input, tf.transpose(K))  # (batch, T-1, d)

        # --- Multi-step rollout ---
        H = min(self.rollout_steps, 5)  # cap for memory
        # Compute K^k for k = 1..H
        K_powers = [K]  # K^1
        for k in range(1, H):
            if is_batched_K:
                K_powers.append(tf.matmul(K_powers[-1], K))
            else:
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
            if is_batched_K:
                z_pred_k = tf.einsum("btj,bji->bti", z_start, tf.transpose(K_powers[k_idx], perm=[0, 2, 1]))
            else:
                z_pred_k = tf.matmul(z_start, tf.transpose(K_powers[k_idx]))
            
            z_true_k = z_sequence[:, k:max_start + 1 + k, :]     # (batch, max_start+1, d)
            multi_preds.append(z_pred_k)
            multi_targets.append(z_true_k)

        # Stack along rollout dimension: (batch, max_start+1, H, d)
        multi_step_pred = tf.stack(multi_preds, axis=2)
        multi_step_target = tf.stack(multi_targets, axis=2)

        # --- Eigenvalue decomposition (for spectral analysis/loss) ---
        # Cast K to float32 for eigvals
        K_f32 = tf.cast(K, tf.float32)
        if is_batched_K:
            # Fast O(1) batched eigenvalues for spectral features:
            # computing eigvals on batch=128 of 128x128 matrices is very slow.
            # We subsample 2 matrices max, compute eigs, and broadcast.
            K_SUBSAMPLE = 2
            # slice the first K_SUBSAMPLE
            K_sub = K_f32[:K_SUBSAMPLE]
            eigenvalues = tf.linalg.eigvals(K_sub) # (min(batch, 2), d)
            # tile to match batch size
            multiples = tf.cast(tf.math.ceil(batch_size / tf.shape(eigenvalues)[0]), tf.int32)
            eigenvalues = tf.tile(eigenvalues, [multiples, 1])[:batch_size]
        else:
            eigenvalues = tf.linalg.eigvals(K_f32)  # (d,) complex

        # --- Final latent state for RUL head ---
        final_state = z_sequence[:, -1, :]  # (batch, d)

        outputs = {
            "one_step_pred": z_pred_one,
            "one_step_target": z_target,
            "multi_step_pred": multi_step_pred,
            "multi_step_target": multi_step_target,
            "K": K,
            "eigenvalues": eigenvalues,
            "final_state": final_state,
            "U": self.U,
            "V": self.V,
        }
        if sigma is not None:
            outputs["sigma"] = sigma
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
            "condition_dim": self.condition_dim,
            "conditioning_strategy": self.conditioning_strategy,
            "condition_indices": self.condition_indices,
            "kmeans_centroids": self.kmeans_centroids.tolist() if self.kmeans_centroids is not None else None,
            "condition_net_type": self.condition_net_type,
        })
        return config


# =========================================================================
# Spectral Feature Extraction (functional helper)
# =========================================================================

def extract_spectral_features(eigenvalues, top_k: int = 4):
    """Extract interpretable spectral features from Koopman eigenvalues.

    Args:
        eigenvalues: (d,) or (batch, d) complex tensor — eigenvalues of K
        top_k: number of dominant modes to extract features from

    Returns:
        spectral_features: (top_k * 2 + 2,) or (batch, top_k * 2 + 2) float tensor
    """
    is_batched = len(eigenvalues.shape) == 2
    if not is_batched:
        eigenvalues = tf.expand_dims(eigenvalues, 0)  # (1, d)

    eig_mags = tf.abs(eigenvalues)   # (batch, d)

    # Sort by magnitude (descending)
    sorted_indices = tf.argsort(eig_mags, direction="DESCENDING")
    
    # We need batched gather: tf.gather(..., batch_dims=1)
    top_indices = sorted_indices[:, :top_k]
    top_eigs = tf.gather(eigenvalues, top_indices, batch_dims=1)  # (batch, top_k)
    
    # Complex logarithm: log(λ) = log|λ| + i·arg(λ)
    log_eigs = tf.math.log(tf.cast(top_eigs, tf.complex64) + 1e-10)
    decay_rates = -tf.math.real(log_eigs)      # (batch, top_k)
    frequencies = tf.abs(tf.math.imag(log_eigs))  # (batch, top_k)

    # Global spectral features
    spectral_radius = tf.reduce_max(eig_mags, axis=-1, keepdims=True)  # (batch, 1)
    all_mags_sorted = tf.sort(eig_mags, direction="DESCENDING", axis=-1)
    spectral_gap = tf.expand_dims(all_mags_sorted[:, 0] - all_mags_sorted[:, 1], -1) # (batch, 1)

    # Concatenate: [σ_1, ..., σ_k, ω_1, ..., ω_k, ρ, gap] along axis=1
    spectral_feats = tf.concat([
        tf.cast(decay_rates, tf.float32),
        tf.cast(frequencies, tf.float32),
        tf.cast(spectral_radius, tf.float32),
        tf.cast(spectral_gap, tf.float32),
    ], axis=1)  # (batch, top_k * 2 + 2)

    if not is_batched:
        spectral_feats = tf.squeeze(spectral_feats, 0)
        
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
