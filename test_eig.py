import time
import tensorflow as tf
import os

# Limit visible devices to test on CPU/GPU
tf.config.optimizer.set_jit(True)

d = 128
batch = 1024

K = tf.Variable(tf.random.normal((batch, d, d)))

with tf.GradientTape() as tape:
    start = time.time()
    eigs = tf.linalg.eigvals(K)
    loss = tf.reduce_sum(tf.abs(eigs))
    
start_bwd = time.time()
grads = tape.gradient(loss, K)
print(f"Batched (1024x128x128) with grad: FWD {start_bwd - start:.4f}s, BWD {time.time() - start_bwd:.4f}s")

K_nograd = tf.stop_gradient(K)
start = time.time()
eigs2 = tf.linalg.eigvals(K_nograd)
print(f"Batched (1024x128x128) no grad: {time.time() - start:.4f}s")

K_single = tf.random.normal((d, d))
start = time.time()
eigs_s = tf.linalg.eigvals(K_single)
print(f"Single (128x128) no grad: {time.time() - start:.4f}s")
