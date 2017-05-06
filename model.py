import numpy as np
import tensorflow as tf
import random


def batch_norm(shape, x, phase_train, scope='BN'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the order of the input tensor
    
    Args:
        shape:       Tuple, shape of input
        x:           Tensor,  B...D input maps (e.g. BHWD or BXYZD)
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


def segnet_graph(shape, beta=0.01, seed=None, load_model=None):
    if load_model is None:
        pass
    x_shape = shape  # 1st dim should be the size of dataset
    y_shape = shape
    y_shape[-1] = 1  # All but last dim should be same shape as x_shape

    with tf.name_scope('Input') as scope:
        x = tf.placeholder(tf.int32, shape=x_shape, name="X")
        y = tf.placeholder(tf.int32, shape=y_shape, name="Y")
        phase_train = tf.placeholder(tf.bool, name="Phase")

    with tf.name_scope('Preprocessing') as scope:
        # We want to normalize
        x_norm = batch_norm(x_shape, x, phase_train, scope=scope)

    # Start the rest of the actual network
    pass
