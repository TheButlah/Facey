import numpy as np
import tensorflow as tf
import random

from util import nearest_neighbor_2d, nearest_neighbor_3d


def batch_norm(x, shape, phase_train, scope='BN'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the order of the input tensor, where 1<=n<=3
    
    Args:
        x:           Tensor,  B...D input maps (e.g. BHWD or BXYZD)
        shape:       Tuple, shape of input
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope

    Returns:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        n_out = shape[-1]  # depth of input maps
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, shape[:-1], name='moments')
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


def conv(input, input_shape, num_features, phase_train, size=3, seed=None, scope='Conv'):
    with tf.variable_scope(scope):
        kernel_shape = [size]*(len(input_shape)-2)
        kernel_shape.append(input_shape[-1])
        kernel_shape.append(num_features)
        # example: input_shape is BHWD, kernel_shape is [3,3,D,num_features]
        kernel = tf.Variable(tf.random_normal(kernel_shape, seed=seed, name='Kernel'))
        convolved = tf.nn.convolution(input, kernel, padding="SAME", name='Conv')
        convolved_shape = input_shape
        convolved_shape[-1] = num_features
        # example: input_shape is BHWD, convolved_shape is [B,H,W,num_features]
        return batch_norm(convolved, convolved_shape, phase_train), convolved_shape


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


def setup_graph(shape, beta=0.01, seed=None, load_model=None):
    if load_model is None:
        pass
    x_shape = shape  # 1st dim should be the size of dataset
    y_shape = shape
    y_shape[-1] = 1  # All but last dim should be same shape as x_shape

    with tf.variable_scope('Input'):
        x = tf.placeholder(tf.int32, shape=x_shape, name="X")
        y = tf.placeholder(tf.int32, shape=y_shape, name="Y")
        phase_train = tf.placeholder(tf.bool, name="Phase")

    with tf.variable_scope('Preprocessing'):
        # We want to normalize
        x_norm = batch_norm(x, x_shape, phase_train, scope='X-Norm')

    with tf.variable_scope('Encoder'):
        conv1_1, last_shape = conv(x_norm, x_shape, 64, phase_train, seed=seed, scope='Conv1_1')
        relu1_1 = relu(conv1_1, scope='Relu1_1')
        conv1_2, last_shape = conv(relu1_1, last_shape, 64, phase_train, seed=seed, scope='Conv1_2')
        relu1_2 = relu(conv1_2, scope='Relu1_2')
        pool1, last_shape, mask1 = pool(relu1_2, last_shape, scope='Pool1')

        conv2_1, last_shape = conv(pool1, last_shape, 64, phase_train, seed=seed, scope='Conv2_1')
        relu2_1 = relu(conv2_1, scope='Relu2_1')
        conv2_2, last_shape = conv(relu2_1, last_shape, 64, phase_train, seed=seed, scope='Conv2_2')
        relu2_2 = relu(conv2_2, scope='Relu2_2')
        pool2, last_shape, mask2 = pool(relu2_2, last_shape, scope='Pool2')

        conv3_1, last_shape = conv(pool2, last_shape, 64, phase_train, seed=seed, scope='Conv3_1')
        relu3_1 = relu(conv3_1, scope='Relu3_1')
        conv3_2, last_shape = conv(relu3_1, last_shape, 64, phase_train, seed=seed, scope='Conv3_2')
        relu3_2 = relu(conv3_2, scope='Relu3_2')
        pool3, last_shape, mask3 = pool(relu3_2, last_shape, scope='Pool3')

        conv4_1, last_shape = conv(pool3, last_shape, 64, phase_train, seed=seed, scope='Conv4_1')
        relu4_1 = relu(conv4_1, scope='Relu4_1')
        conv4_2, last_shape = conv(relu4_1, last_shape, 64, phase_train, seed=seed, scope='Conv4_2')
        relu4_2 = relu(conv4_2, scope='Relu4_2')
        pool4, last_shape, mask4 = pool(relu4_2, last_shape, scope='Pool4')

        conv5_1, last_shape = conv(pool4, last_shape, 64, phase_train, seed=seed, scope='Conv5_1')
        relu5_1 = relu(conv5_1, scope='Relu5_1')
        conv5_2, last_shape = conv(relu5_1, last_shape, 64, phase_train, seed=seed, scope='Conv5_2')
        relu5_2 = relu(conv5_2, scope='Relu5_2')
        pool5, last_shape, mask5 = pool(relu5_2, last_shape, scope='Pool5')

    with tf.variable_scope('Decoder'):
        pass
    # Start the rest of the actual network
    pass
