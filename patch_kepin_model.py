import sys

def patch_model_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Modify KePINModel.__init__
    old_init = """    def __init__(self, input_shape_tuple, arch_config=None, n_train=None,
                 n_active_losses=7, condition_dim=0, condition_indices=None, **kwargs):
        super().__init__(**kwargs)"""
    new_init = """    def __init__(self, input_shape_tuple, arch_config=None, n_train=None,
                 n_active_losses=7, condition_dim=0, condition_indices=None, 
                 conditioning_strategy="raw3", kmeans_centroids=None, **kwargs):
        super().__init__(**kwargs)"""
    content = content.replace(old_init, new_init)

    # Add variables to init
    old_vars = """        self.condition_dim = condition_dim
        self.condition_indices = condition_indices"""
    new_vars = """        self.condition_dim = condition_dim
        self.condition_indices = condition_indices
        self.conditioning_strategy = conditioning_strategy
        self.kmeans_centroids = kmeans_centroids"""
    content = content.replace(old_vars, new_vars)

    # 2. Update KoopmanOperator call in __init__
    old_koopman = """        self.koopman = KoopmanOperator(
            latent_dim=arch_config["latent_dim"],
            rollout_steps=arch_config["rollout"],
            stability_mode="svd",
            condition_dim=self.condition_dim,
            name="koopman_operator",
        )"""
    new_koopman = """        self.koopman = KoopmanOperator(
            latent_dim=arch_config["latent_dim"],
            rollout_steps=arch_config["rollout"],
            stability_mode="svd",
            condition_dim=self.condition_dim,
            conditioning_strategy=self.conditioning_strategy,
            condition_indices=self.condition_indices,
            kmeans_centroids=self.kmeans_centroids,
            name="koopman_operator",
        )"""
    content = content.replace(old_koopman, new_koopman)

    # 3. Remove _extract_condition entirely
    # It starts at `    def _extract_condition(self, inputs):` and ends before `    def call(self, inputs, training=None):`
    start_idx = content.find("    def _extract_condition(self, inputs):")
    end_idx = content.find("    def call(self, inputs, training=None):")
    content = content[:start_idx] + content[end_idx:]

    # 4. Update call method
    old_call = """    def call(self, inputs, training=None):
        x = inputs
        mu = self._extract_condition(inputs)

        # Input projection"""
    new_call = """    def call(self, inputs, training=None):
        x = inputs

        # Input projection"""
    content = content.replace(old_call, new_call)

    old_koop_call = """        # Koopman projection & rollout
        koopman_out = self.koopman(z_sequence, condition_vector=mu)"""
    new_koop_call = """        # Koopman projection & rollout
        koopman_out = self.koopman(z_sequence, raw_inputs=inputs)"""
    content = content.replace(old_koop_call, new_koop_call)

    # 5. Update build_kepin_model
    old_build = """def build_kepin_model(seq_len, n_features, n_train=None, arch_config=None,
                      n_active_losses=7, condition_dim=0, condition_indices=None):
    return KePINModel(
        input_shape_tuple=(seq_len, n_features),
        arch_config=arch_config, n_train=n_train,
        n_active_losses=n_active_losses,
        condition_dim=condition_dim,
        condition_indices=condition_indices,
    )"""
    new_build = """def build_kepin_model(seq_len, n_features, n_train=None, arch_config=None,
                      n_active_losses=7, condition_dim=0, condition_indices=None,
                      conditioning_strategy="raw3", kmeans_centroids=None):
    return KePINModel(
        input_shape_tuple=(seq_len, n_features),
        arch_config=arch_config, n_train=n_train,
        n_active_losses=n_active_losses,
        condition_dim=condition_dim,
        condition_indices=condition_indices,
        conditioning_strategy=conditioning_strategy,
        kmeans_centroids=kmeans_centroids,
    )"""
    content = content.replace(old_build, new_build)

    with open(file_path, 'w') as f:
        f.write(content)

patch_model_file("kepin_model.py")
