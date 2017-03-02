import tensorflow as tf
import util

# Encoder network, generates low res feature map with intermediate max pooling
# indicies
def encode(x, f, k, N):
    h = tf.Variable(tf.random_normal([f, f, 3, k]))
    y = tf.nn.convolution(x, h, "SAME")
    for i in range(N):
        y = util.down_sample(y, f)
    return y

# Decoder network, generates high res output map using pooling indices for
# upsampling
def decode(x, idxs, f, k):
    h = tf.Variable(tf.random_normal([f, f, k, 3]))
    y = x
    for i in range(N):
        y = util.up_sample(y, idxs[i], f)
    return tf.nn.convolution(y, h, "SAME")
