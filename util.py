import tensorflow as tf
import numpy as np


def batch_norm(x, shape, phase_train, scope='BN'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the spatial dimensions of the input tensor, where 1<=d<=3

    Args:
        x:           Tensor,  B...D input maps (e.g. BHWC or BXYZC)
        shape:       Tuple, shape of input
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope

    Returns:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        n_out = shape[-1]  # depth of input maps
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='Beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='Gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, list(range(len(shape[:-1]))), name='Moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def conv(input, input_shape, num_features, phase_train, do_bn=True, size=3, seed=None, scope='Conv'):
    with tf.variable_scope(scope):
        kernel_shape = [size]*(len(input_shape)-2)
        kernel_shape.append(input_shape[-1])
        kernel_shape.append(num_features)
        # example: input_shape is BHWD, kernel_shape is [3,3,D,num_features]
        kernel = tf.Variable(tf.random_normal(kernel_shape, seed=seed, name='Kernel'))
        convolved = tf.nn.convolution(input, kernel, padding="SAME", name='Conv')
        convolved_shape = list(input_shape)
        convolved_shape[-1] = num_features
        # example: input_shape is BHWD, convolved_shape is [B,H,W,num_features]
        if do_bn:
            return batch_norm(convolved, convolved_shape, phase_train), convolved_shape
        else:
            return convolved, convolved_shape


def relu(input, scope='Relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(input, name='Relu')


def pool(input, input_shape, scope='Pool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = tf.nn.pool(input, window_shape=window_shape, pooling_type="MAX", strides=window_shape, padding="SAME")
        output_shape = [input_shape[0]] + [i / 2 for i in input_shape[1:-1]] + [input_shape[-1]]
        mask = nearest_neighbor(output)
        mask = tf.equal(input, mask)
        mask = tf.cast(mask, tf.float32)
        return output, output_shape, mask


def unpool(input, input_shape, mask, scope='Unpool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = nearest_neighbor(input) * mask
        output_shape = [input_shape[0]] + [i*2 for i in input_shape[1:-1]] + [input_shape[-1]]
        return output, output_shape


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
    print(indices)
    for row in indices:
        print(row)
        if np.sum(row >= np.zeros([1, 3])) == 3 and np.sum(row < divisions) == 3:
            print("hi")
            output[row[0], row[1], row[2]] = 1
    return output

# lower_left = np.array([1.0, 0.0, 0.0])
# upper_right = np.array([4.0, 3.0, 6.0])
# division = np.array([6, 6, 12])
# input = np.array([[1.1, 1.2, 1.3], [1.1, 1.2, 5.2], [1.2, 1.1, 3.01], [4.1, 2.2, 5.2]])
# output = gen_occupancy_grid(input, lower_left, upper_right, division)
# print output

'''
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
    return tf.einsum('ijkl,jkml->ijkm', x, h)'''
