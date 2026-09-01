import tensorflow as tf
import keras
from keras.saving import register_keras_serializable

@register_keras_serializable(package="KePIN")
class KANLayer(keras.layers.Layer):
    """Simple efficient Kolmogorov-Arnold Network (KAN) Layer using SiLU bases."""
    def __init__(self, output_dim, grid_size=5, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.grid_size = grid_size
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.grid = self.add_weight(name="grid", shape=(input_dim, self.grid_size),
                                    initializer=keras.initializers.RandomUniform(minval=-2., maxval=2.),
                                    trainable=True)
        self.w = self.add_weight(name="w", shape=(input_dim, self.grid_size, self.output_dim),
                                 initializer="glorot_uniform", trainable=True)
        self.base_w = self.add_weight(name="base_w", shape=(input_dim, self.output_dim),
                                      initializer="glorot_uniform", trainable=True)
        self.base_bias = self.add_weight(name="base_bias", shape=(self.output_dim,),
                                         initializer="zeros", trainable=True)
        super().build(input_shape)
        
    def call(self, x):
        # x: (batch, input_dim)
        x_exp = tf.expand_dims(x, -1) # (batch, input_dim, 1)
        grid_exp = tf.expand_dims(self.grid, 0) # (1, input_dim, grid_size)
        basis = tf.nn.silu(x_exp - grid_exp) # (batch, input_dim, grid_size)
        
        # Sum over basis and input dimensions
        # basis: [B, I, G], w: [I, G, O] -> [B, O]
        spline_out = tf.einsum('big,igo->bo', basis, self.w)
        
        # Base linear + silu component
        base_out = tf.matmul(tf.nn.silu(x), self.base_w)
        
        return spline_out + base_out + self.base_bias

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "grid_size": self.grid_size,
        })
        return config
