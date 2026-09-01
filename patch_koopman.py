import sys

def patch_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Insert Strategy Classes
    strategies_code = """
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
        self.centroids_var = tf.constant(self.kmeans_centroids, dtype=tf.float32)
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
class KoopmanOperator(keras.layers.Layer):"""
    
    content = content.replace('@register_keras_serializable(package="KePIN")\nclass KoopmanOperator(keras.layers.Layer):', strategies_code)
    
    # 2. Modify __init__
    old_init = """    def __init__(self, latent_dim: int, rollout_steps: int = 3,
                 stability_mode: str = "svd", condition_dim: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode
        self.condition_dim = condition_dim"""
    new_init = """    def __init__(self, latent_dim: int, rollout_steps: int = 3,
                 stability_mode: str = "svd", condition_dim: int = 0,
                 conditioning_strategy: str = "raw3",
                 condition_indices: list = None,
                 kmeans_centroids = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.rollout_steps = rollout_steps
        self.stability_mode = stability_mode
        self.condition_dim = condition_dim
        self.conditioning_strategy = conditioning_strategy
        self.condition_indices = condition_indices
        self.kmeans_centroids = kmeans_centroids"""
    content = content.replace(old_init, new_init)

    # 3. Modify build
    old_build = """            # Add parameter-conditioning network if needed
            if self.condition_dim > 0:
                self.condition_net = keras.Sequential([
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(
                        d, 
                        kernel_initializer="zeros", 
                        bias_initializer="zeros",
                        name="delta_s_out"
                    )
                ], name="condition_net")"""
    new_build = """            # Add parameter-conditioning network if needed
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
                
                # 2. Setup MLP
                self.condition_net = keras.Sequential([
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dense(
                        d, 
                        kernel_initializer="zeros", 
                        bias_initializer="zeros",
                        name="delta_s_out"
                    )
                ], name="condition_net")"""
    content = content.replace(old_build, new_build)

    # 4. Modify _get_K
    old_get_K = """    def _get_K(self, mu=None):
        \"\"\"Construct the Koopman operator matrix K.
        If mu is provided (batch, condition_dim), K will be (batch, d, d).
        Otherwise K is (d, d).
        \"\"\"
        if self.stability_mode == "svd":
            s_val = self.s
            if self.condition_dim > 0 and mu is not None:
                # Add conditional perturbation: s(mu) = s + delta_s(mu)
                delta_s = self.condition_net(mu) # (batch, d)"""
    new_get_K = """    def _get_K(self, raw_inputs=None):
        \"\"\"Construct the Koopman operator matrix K.
        If raw_inputs is provided, we compute mu = strategy(raw_inputs).
        \"\"\"
        if self.stability_mode == "svd":
            s_val = self.s
            mu = None
            if self.condition_dim > 0 and raw_inputs is not None:
                mu = self.strategy_layer(raw_inputs)
                # Add conditional perturbation: s(mu) = s + delta_s(mu)
                delta_s = self.condition_net(mu) # (batch, d)"""
    content = content.replace(old_get_K, new_get_K)
    
    # 5. Modify call definition and usage
    old_call = """    def call(self, z_sequence, condition_vector=None, training=None):
        \"\"\"Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            condition_vector: (batch, condition_dim) optional
        \"\"\"
        K, sigma = self._get_K(mu=condition_vector)"""
    new_call = """    def call(self, z_sequence, raw_inputs=None, training=None):
        \"\"\"Forward pass.

        Args:
            z_sequence: (batch, T, d) — sequence of latent states
            raw_inputs: (batch, T, n_features) raw inputs to derive condition
        \"\"\"
        K, sigma = self._get_K(raw_inputs=raw_inputs)"""
    content = content.replace(old_call, new_call)
    
    # 6. get_config
    old_config = """    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
            "condition_dim": self.condition_dim,
        })
        return config"""
    new_config = """    def get_config(self):
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "rollout_steps": self.rollout_steps,
            "stability_mode": self.stability_mode,
            "condition_dim": self.condition_dim,
            "conditioning_strategy": self.conditioning_strategy,
            "condition_indices": self.condition_indices,
            "kmeans_centroids": self.kmeans_centroids.tolist() if self.kmeans_centroids is not None else None,
        })
        return config"""
    content = content.replace(old_config, new_config)

    with open(file_path, 'w') as f:
        f.write(content)

patch_file("koopman_module.py")
