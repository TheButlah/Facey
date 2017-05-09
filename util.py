import tensorflow as tf
import numpy as np


# Determine number of channels from tensor of shape [height, width, channels]
def get_channels(x):
    return x.get_shape().as_list()[-1]


# Create a random filter of size [f, f, k, k]
def random_filter(f, k):
    return tf.Variable(tf.random_normal([f, f, f, k, k]))


# Batch normalize
def batch_norm(x):
    mu, sigma = tf.nn.moments(x, [0])
    return tf.nn.batch_normalization(x, mu, sigma, 0, 1, 1e-6)


# Convolve tensor of shape [n, n, k] with filter [f, f, k, k]
# Maintains size and number of channels
# Also performs batch norm and relu
def conv(x, f, k):
    h = random_filter(f, k)
    y = tf.nn.convolution(x, h, "SAME")
    y = batch_norm(y)
    return tf.nn.relu(y)


# Max pool
def down_sample(x, f):
    k = get_channels(x)
    x = conv(x, f, k)
    y = tf.nn.pool(x, window_shape=[2, 2, 2], pooling_type="MAX", strides=[2, 2, 2], padding="SAME")
    print(y.get_shape())
    mask = tf.equal(x, nearest_neighbor_3d(y))
    mask = tf.cast(mask, tf.float32)
    return y, mask


# Create sparse higher-res tensor using pooling indices
def up_sample(x, mask, f):
    k = get_channels(x)
    y = nearest_neighbor_3d(x)
    y = y * mask
    return conv(y, f, k)


# Linear transformation from tensor of shape [num_batches, n, n, k] to
# [num_batches, n, n, num_classes]
# h has shape [n, n, k, num_classes]
def tensor_mul(x, h):
    return tf.einsum('ijkl,jkml->ijkm', x, h)


def nearest_neighbor_2d(x):
    s = x.get_shape().as_list()
    n = s[1]
    c = s[-1]
    y = tf.tile(x, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n * n, 1, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n, 2 * n, c])
    return y


def nearest_neighbor_3d(x):
    s = x.get_shape().as_list()
    n = s[1]
    c = s[-1]
    y = tf.transpose(x, [0, 3, 1, 2, 4])
    y = tf.reshape(y, [-1, n, n * n, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n * n, n, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 4 * n * n * n, 1, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * n, 2 * n, 2 * n, c])
    y = tf.transpose(y, [0, 2, 3, 1, 4])
    return y


def gen_occupancy_grid(input, lower_left, upper_right, divisions):
    output = np.zeros(divisions)
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = input - lower_left
    indices = np.floor(offsets / intervals)
    indices = indices.astype(int)
    print indices
    for row in indices:
        print row
        if np.sum(row >= np.zeros([1, 3])) == 3 and np.sum(row < divisions) == 3:
            print "hi"
            output[row[0], row[1], row[2]] = 1
    return output

# lower_left = np.array([1.0, 0.0, 0.0])
# upper_right = np.array([4.0, 3.0, 6.0])
# division = np.array([6, 6, 12])
# input = np.array([[1.1, 1.2, 1.3], [1.1, 1.2, 5.2], [1.2, 1.1, 3.01], [4.1, 2.2, 5.2]])
# output = gen_occupancy_grid(input, lower_left, upper_right, division)
# print output
