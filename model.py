import numpy as np
import tensorflow as tf
import random

from time import time
from util import *


class GenSeg:

    def __init__(self, input_shape, num_classes, seed=None, load_model=None):
        print("Constructing Architecture...")
        self._input_shape = tuple(input_shape)
        x_shape = tuple(input_shape)  # 1st dim should be the size of dataset
        y_shape = tuple(input_shape[:-1])  # Rank of y should be one less
        self._num_classes = num_classes
        self._seed = seed
        self._graph = tf.Graph()
        self._vars = []
        with self._graph.as_default():
            with tf.variable_scope('Input'):
                self._x = tf.placeholder(tf.float32, shape=x_shape, name="X")
                self._y = tf.placeholder(tf.int32, shape=y_shape, name="Y")
                self._phase_train = tf.placeholder(tf.bool, name="Phase")

            with tf.variable_scope('Preprocessing'):
                # We want to normalize
                x_norm = batch_norm(self._x, x_shape, self._phase_train, scope='X-Norm')

            with tf.variable_scope('Encoder'):
                conv1_1, last_shape = conv(x_norm, x_shape, 64, self._phase_train, seed=seed, scope='Conv1_1')
                relu1_1 = relu(conv1_1, scope='Relu1_1')
                conv1_2, last_shape = conv(relu1_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv1_2')
                relu1_2 = relu(conv1_2, scope='Relu1_2')
                pool1, last_shape, mask1 = pool(relu1_2, last_shape, scope='Pool1')

                conv2_1, last_shape = conv(pool1, last_shape, 64, self._phase_train, seed=seed, scope='Conv2_1')
                relu2_1 = relu(conv2_1, scope='Relu2_1')
                conv2_2, last_shape = conv(relu2_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv2_2')
                relu2_2 = relu(conv2_2, scope='Relu2_2')
                pool2, last_shape, mask2 = pool(relu2_2, last_shape, scope='Pool2')

                conv3_1, last_shape = conv(pool2, last_shape, 64, self._phase_train, seed=seed, scope='Conv3_1')
                relu3_1 = relu(conv3_1, scope='Relu3_1')
                conv3_2, last_shape = conv(relu3_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv3_2')
                relu3_2 = relu(conv3_2, scope='Relu3_2')
                pool3, last_shape, mask3 = pool(relu3_2, last_shape, scope='Pool3')

                conv4_1, last_shape = conv(pool3, last_shape, 64, self._phase_train, seed=seed, scope='Conv4_1')
                relu4_1 = relu(conv4_1, scope='Relu4_1')
                conv4_2, last_shape = conv(relu4_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv4_2')
                relu4_2 = relu(conv4_2, scope='Relu4_2')
                pool4, last_shape, mask4 = pool(relu4_2, last_shape, scope='Pool4')

            with tf.variable_scope('Decoder'):
                unpool5, last_shape = unpool(pool4, last_shape, mask4, scope='Unpool5')
                conv5_1, last_shape = conv(unpool5, last_shape, 64, self._phase_train, seed=seed, scope='Conv5_1')
                conv5_2, last_shape = conv(conv5_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv5_2')

                unpool6, last_shape = unpool(conv5_2, last_shape, mask3, scope='Unpool6')
                conv6_1, last_shape = conv(unpool6, last_shape, 64, self._phase_train, seed=seed, scope='Conv6_1')
                conv6_2, last_shape = conv(conv6_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv6_2')

                unpool7, last_shape = unpool(conv6_2, last_shape, mask2, scope='Unpool7')
                conv7_1, last_shape = conv(unpool7, last_shape, 64, self._phase_train, seed=seed, scope='Conv7_1')
                conv7_2, last_shape = conv(conv7_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv7_2')

                unpool8, last_shape = unpool(conv7_2, last_shape, mask1, scope='Unpool8')
                conv8_1, last_shape = conv(unpool8, last_shape, 64, self._phase_train, seed=seed, scope='Conv8_1')
                conv8_2, last_shape = conv(conv8_1, last_shape, 64, self._phase_train, seed=seed, scope='Conv8_2')

            with tf.variable_scope('Softmax'):
                scores, _ = conv(conv8_2, last_shape, num_classes, self._phase_train, do_bn=False, seed=seed, scope='Scores')
                self._y_hat = tf.nn.softmax(scores, name='Y-Hat')  # Operates on last dimension

            with tf.variable_scope('Pipelining'):
                self._loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=self._y),
                    name='Loss'
                )
                self._train_step = tf.train.AdamOptimizer().minimize(self._loss)

            self._sess = tf.Session(graph=self._graph)  # Not sure if this actually needs to specify the graph
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, y_train, batch_size, num_epochs=1000):
        X = np.random.rand(*self._input_shape)  # allows the list to be passed as the various arguments to the function
        Y = np.random.random_integers(0, self._num_classes-1, size=self._input_shape[:-1])

        with self._sess.as_default():
            # Training loop for parameter tuning
            print("Starting training for %d epochs" % num_epochs)
            last_time = time()
            for epoch in range(num_epochs):
                _, loss_val = self._sess.run(
                    [self._train_step, self._loss],
                    feed_dict={self._x: X, self._y: Y, self._phase_train: True}
                )
                current_time = time()
                if (current_time - last_time) >= 5:  # Only print progress every 5 seconds
                    last_time = current_time
                    print("Current Loss Value: %.10f, Percent Complete: %f" % (loss_val, epoch / num_epochs))
            print("Completed Training.")
