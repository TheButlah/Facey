import tensorflow as tf
import util

# Encoder network, generates low res feature map with intermediate max pooling
# indicies
def encode(x, f, k, N):
    h = tf.Variable(tf.random_normal([f, f, f, 1, k]))
    y = tf.nn.convolution(x, h, "SAME")
    masks = []
    for i in range(N):
        y, mask = util.down_sample(y, f)
        masks = [mask] + masks
    return y, masks

# Decoder network, generates high res output map using pooling indices for
# upsampling
def decode(x, masks, f, k):
    h = tf.Variable(tf.random_normal([f, f, k, 3]))
    y = x
    for mask in masks:
        y = util.up_sample(y, mask, f)
    return tf.nn.convolution(y, h, "SAME")
