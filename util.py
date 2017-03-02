import tensorflow as tf

# Determine number of channels from tensor of shape [height, width, channels]
def get_channels(x):
    return x.get_shape().as_list()[-1]

# Create a random filter of size [f, f, k, k]
def random_filter(f, k):
    return tf.Variable(tf.random_normal([f, f, k, k]))

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
    # TODO: Obtain indices from max pool step
    k = get_channels(x)
    y = conv(x, f, k)
    return tf.nn.max_pool(y, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

# Create sparse higher-res tensor using pooling indices
def up_sample(x, idx, f):
    # TODO: Fix upsampling step
    k = get_channels(x)
    y = tf.tile(x, [1, 2, 2, 1])
    return conv(x, f, k)

# Linear transformation from tensor of shape [n, n, 3] to [n, n, num_classes]
# h has shape [3, num_classes]
def tensor_mul(x, h):
    return tf.einsum('ijk,kl->ijl', x, h)
