import tensorflow as tf
import numpy as np

from time import time, strftime
from util import *
import os


class Facey:
    """Facey is an unsupervised convolutional autoencoder for facial reconstruction

    GenSeg is a generalized version of SegNet, in that it can operate on data with N spatial dimensions instead of
    just 2 like in the original SegNet. However, due to restrictions in TensorFlow, specifically in the way that
    convolutions work, this implementation of GenSeg only works for 1<=N<=3 spatial dimensions. This does mean that
    this implementation can handle 3D and 1D data as well as the more conventional 2D data. Additionally, this
    implementation is designed so that once support for N>3 is added into TensorFlow, it should be trivial to add
    that support into this implementation.
    """

    def __init__(self, input_shape, seed=None, load_model=None):
        """Initializes the architecture of GenSeg and returns an instance.

        Currently, only 1<=N<=3 spatial dimensions in the input data are supported due to limitations in TensorFlow, so
        ensure that the input_shape specified follows this restriction.

        Args:
            input_shape:    A list that represents the shape of the input. Can contain None as the first element to
                            indicate that the batch size can vary (this is the preferred way to do it). Example:
                            [None, 32, 32, 32, 1] for 3D data.
            seed:           An integer used to seed the initial random state. Can be None to generate a new random seed.
            load_model:     If not None, then this should be a string indicating the checkpoint file containing data
                            that will be used to initialize the parameters of the model. Typically used when loading a
                            pre-trained model, or resuming a previous training session.
        """
        print("Constructing Architecture...")
        self._input_shape = tuple(input_shape)  # Tuples are used to ensure the dimensions are immutable
        x_shape = tuple(input_shape)  # 1st dim should be the size of dataset
        self._seed = seed
        self._graph = tf.Graph()
        with self._graph.as_default():
            num_features = 32

            with tf.variable_scope('Input'):
                self._x = tf.placeholder(tf.float32, shape=x_shape, name="X")
                self._phase_train = tf.placeholder(tf.bool, name="Phase")

            with tf.variable_scope('Preprocessing'):
                # We want to normalize
                x_norm = batch_norm(self._x, x_shape, self._phase_train, scope='X-Norm')

            with tf.variable_scope('Encoder'):
                conv1_1, last_shape = conv(x_norm, x_shape, num_features, self._phase_train, seed=seed, scope='Conv1_1')
                relu1_1 = relu(conv1_1, scope='Relu1_1')
                conv1_2, last_shape = conv(relu1_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv1_2')
                relu1_2 = relu(conv1_2, scope='Relu1_2')
                pool1, last_shape, mask1 = pool(relu1_2, last_shape, scope='Pool1')

                conv2_1, last_shape = conv(pool1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv2_1')
                relu2_1 = relu(conv2_1, scope='Relu2_1')
                conv2_2, last_shape = conv(relu2_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv2_2')
                relu2_2 = relu(conv2_2, scope='Relu2_2')
                pool2, last_shape, mask2 = pool(relu2_2, last_shape, scope='Pool2')

                conv3_1, last_shape = conv(pool2, last_shape, num_features, self._phase_train, seed=seed, scope='Conv3_1')
                relu3_1 = relu(conv3_1, scope='Relu3_1')
                conv3_2, last_shape = conv(relu3_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv3_2')
                relu3_2 = relu(conv3_2, scope='Relu3_2')
                pool3, last_shape, mask3 = pool(relu3_2, last_shape, scope='Pool3')

                conv4_1, last_shape = conv(pool3, last_shape, num_features, self._phase_train, seed=seed, scope='Conv4_1')
                relu4_1 = relu(conv4_1, scope='Relu4_1')
                conv4_2, last_shape = conv(relu4_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv4_2')
                relu4_2 = relu(conv4_2, scope='Relu4_2')
                pool4, last_shape, mask4 = pool(relu4_2, last_shape, scope='Pool4')

            with tf.variable_scope('Decoder'):
                unpool5, last_shape = unpool(pool4, last_shape, mask4, scope='Unpool5')
                conv5_1, last_shape = conv(unpool5, last_shape, num_features, self._phase_train, seed=seed, scope='Conv5_1')
                conv5_2, last_shape = conv(conv5_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv5_2')

                unpool6, last_shape = unpool(conv5_2, last_shape, mask3, scope='Unpool6')
                conv6_1, last_shape = conv(unpool6, last_shape, num_features, self._phase_train, seed=seed, scope='Conv6_1')
                conv6_2, last_shape = conv(conv6_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv6_2')

                unpool7, last_shape = unpool(conv6_2, last_shape, mask2, scope='Unpool7')
                conv7_1, last_shape = conv(unpool7, last_shape, num_features, self._phase_train, seed=seed, scope='Conv7_1')
                conv7_2, last_shape = conv(conv7_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv7_2')

                unpool8, last_shape = unpool(conv7_2, last_shape, mask1, scope='Unpool8')
                conv8_1, last_shape = conv(unpool8, last_shape, num_features, self._phase_train, seed=seed, scope='Conv8_1')
                conv8_2, last_shape = conv(conv8_1, last_shape, num_features, self._phase_train, seed=seed, scope='Conv8_2')

            with tf.variable_scope('Output'):
                self._x_hat, _ = conv(conv8_2, last_shape, input_shape[-1], self._phase_train, do_bn=False, size=1, seed=seed, scope='Scores')

            with tf.variable_scope('Pipelining'):
                self._loss = tf.nn.l2_loss(self._x - self._x_hat)
                self._train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)

            self._sess = tf.Session(graph=self._graph)  # Not sure if this really needs to explicitly specify the graph
            with self._sess.as_default():
                self._saver = tf.train.Saver()
                if load_model is not None:
                    print("Restoring Model...")
                    load_model = os.path.abspath(load_model)
                    self._saver.restore(self._sess, load_model)
                    print("Model Restored!")
                else:
                    print("Initializing model...")
                    self._sess.run(tf.global_variables_initializer())
                    print("Model Initialized!")

    def train(self, x_train, num_epochs, start_stop_info=True, progress_info=True):
        """Trains the model using the data provided as a batch.

        Because GenSeg typically runs on large datasets, it is often infeasible to load the entire dataset on either
        memory or the GPU. For this reason, the selection of batches is left up to the user, so that s/he can load the
        proper number of data. For best performance, try to make the batch size (size of first dimension) as large as
        possible without exceeding memory to take advantage of the vectorized code that TensorFlow uses.

        That being said, if the entire dataset fits in memory (and GPU memory if using GPU) and mini-batching is not
        desired, then it is preferable to pass the whole dataset to this function and use a higher value for num_epochs.

        Args:
            x_train:  A numpy ndarray that contains the data to train over. Should should have a shape of
                [batch_size, spatial_dim1, ... , spatial_dimN, channels]. Only 1<=N<=3 spatial dimensions are supported
                currently. These should correspond to the shape of y_train.

            num_epochs:  The number of iterations over the provided batch to perform until training is considered to be
                complete. If all your data fits in memory and you don't need to mini-batch, then this should be a large
                number (>1000). Otherwise keep this small (<50) so the model doesn't become skewed by the small size of
                the provided mini-batch too quickly.

            start_stop_info:  If true, print when the training begins and ends.

            progress_info:  If true, print what the current loss and percent completion over the course of training.

        Returns:
            The loss value after training
        """
        with self._sess.as_default():
            # Training loop for parameter tuning
            if start_stop_info:
                print("Starting training for %d epochs" % num_epochs)
            last_time = time()
            for epoch in range(num_epochs):
                _, loss_val = self._sess.run(
                    [self._train_step, self._loss],
                    feed_dict={self._x: x_train, self._phase_train: True}
                )
                current_time = time()
                if progress_info and (current_time - last_time) >= 5:  # Only print progress every 5 seconds
                    last_time = current_time
                    print("Current Loss Value: %.10f, Percent Complete: %.4f" % (loss_val, epoch / num_epochs * 100))
            if start_stop_info:
                print("Completed Training.")
            return loss_val

    def apply(self, x_data):
        """Applies the model to the batch of data provided. Typically called after the model is trained.

        Args:
            x_data:  A numpy ndarray of the data to apply the model to. Should have the same shape as the training data.
                Example: x_data.shape is [batch_size, num_features0, 480, 3] for a 640x480 RGB image

        Returns:
            A numpy ndarray of the data, with the last dimension being the class probabilities instead of channels.
            Example: result.shape is [batch_size, 640, 480, 10] for a 640x480 RGB image with 10 target classes
        """
        with self._sess.as_default():
            return self._sess.run(self._x_hat, feed_dict={self._x: x_data, self._phase_train: False})

    def save_model(self, save_path=None):
        """Saves the model in the specified file.

        Args:
            save_path:  The relative path to the file. By default, it is
                saved/GenSeg-Year-Month-Date_Hour-Minute-Second.ckpt
        """
        with self._sess.as_default():
            print("Saving Model")
            if save_path is None:
                save_path = "saved/GenSeg-%s.ckpt" % strftime("%Y-%m-%d_%H-%M-%S")
            dirname = os.path.dirname(save_path)
            if dirname is not '':
                os.makedirs(dirname, exist_ok=True)
            save_path = os.path.abspath(save_path)
            path = self._saver.save(self._sess, save_path)
            print("Model successfully saved in file: %s" % path)

