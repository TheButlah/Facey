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
def decode(x, masks, f, k, num_classes):
    h = tf.Variable(tf.random_normal([f, f, f, k, num_classes]))
    for mask in masks:
        x = util.up_sample(x, mask, f)
    return tf.nn.convolution(x, h, "SAME")
