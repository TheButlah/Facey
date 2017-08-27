from pathlib import Path

import tensorflow as tf
import numpy as np
import os

from scipy import misc, io
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2lab, lab2rgb
from math import sqrt


def batch_norm(x, shape, phase_train, scope='BN', weights=None):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Note: The original author's code has been modified to generalize the spatial dimensions of the input tensor.

    Args:
        x:           Tensor,  B...C input maps (e.g. BHWC or BXYZC)
        shape:       Tuple, shape of input
        phase_train: boolean tf.Variable, true indicates training phase
        scope:       string, variable scope

    Returns:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        n_out = shape[-1]  # depth of input maps
        if weights is None:
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='Beta', trainable=True)
        else:
            beta = tf.Variable(weights[str(scope) + '_b'], name='Beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='Gamma', trainable=True)
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


def conv(x, input_shape, num_features, phase_train, weights=None, do_bn=True, size=3, seed=None, scope='Conv'):
    with tf.variable_scope(scope):
        kernel_shape = [size]*(len(input_shape)-2)
        kernel_shape.append(input_shape[-1])
        kernel_shape.append(num_features)
        # example: input_shape is BHWC, kernel_shape is [3,3,D,num_features]
        # kernel = tf.Variable(tf.random_normal(kernel_shape, seed=seed, name='Kernel'))
        if weights is None:
            kernel = tf.Variable(tf.random_normal(kernel_shape, seed=seed, name='Kernel'))
        else:
            kernel = tf.Variable(weights[scope + '_W'])
        convolved = tf.nn.convolution(x, kernel, padding="SAME", name='Conv')
        convolved_shape = list(input_shape)
        convolved_shape[-1] = num_features
        # example: input_shape is BHWC, convolved_shape is [B,H,W,num_features]
        if do_bn:
            return batch_norm(convolved, convolved_shape, phase_train, scope=scope), convolved_shape
        else:
            return convolved, convolved_shape


def relu(x, scope='Relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(x, name='Relu')


def pool(x, input_shape, scope='Pool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = tf.nn.pool(x, window_shape=window_shape, pooling_type="MAX", strides=window_shape, padding="SAME")
        output_shape = [input_shape[0]] + [i / 2 for i in input_shape[1:-1]] + [input_shape[-1]]
        mask = nearest_neighbor(output)
        mask = tf.equal(x, mask)
        mask = tf.cast(mask, tf.float32)
        return output, output_shape, mask


def unpool(x, input_shape, mask, scope='Unpool'):
    with tf.variable_scope(scope):
        if len(input_shape) == 4:  # 2D
            nearest_neighbor = nearest_neighbor_2d
            window_shape = [2, 2]
        elif len(input_shape) == 5:  # 3D
            nearest_neighbor = nearest_neighbor_3d
            window_shape = [2, 2, 2]
        else:
            raise Exception('Tensor shape not supported')

        output = nearest_neighbor(x) * mask
        output_shape = [input_shape[0]] + [i*2 for i in input_shape[1:-1]] + [input_shape[-1]]
        return output, output_shape


def nearest_neighbor_2d(x):
    s = x.get_shape().as_list()
    h = s[1]
    w = s[2]
    c = s[-1]
    y = tf.tile(x, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * h * w, 1, c])
    y = tf.tile(y, [1, 1, 2, 1])
    y = tf.reshape(y, [-1, 2 * h, 2 * w, c])
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


def gen_occupancy_grid(x, lower_left, upper_right, divisions):
    output = np.zeros(np.append(divisions, 1))
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = x - lower_left
    indices = np.floor(offsets / intervals)
    indices = indices.astype(int)
    for row in indices:
        if np.sum(row >= np.zeros([1, 3])) == 3 and np.sum(row < divisions) == 3:
            output[row[0], row[1], row[2], 0] = 1
    return output


def gen_label_occupancy_grid(x, lower_left, upper_right, divisions, num_classes):
    output = np.zeros(np.append(divisions, num_classes))
    lengths = upper_right - lower_left
    intervals = lengths / divisions
    offsets = x - np.append(lower_left, 0)
    indices = np.floor(offsets / np.append(intervals, 1))
    indices = indices.astype(int)
    for row in indices:
        if np.sum(row[:3] >= np.zeros([1, 3])) == 3 and np.sum(row[:3] < divisions) == 3:
            output[row[0], row[1], row[2], row[3]] += 1
    output = np.argmax(output, -1)
    return output


def image_to_angles(x, y):
    theta = -np.arctan2(y, x)
    phi = np.arctan2(z, x)
    return theta, phi


def velodyne_to_angles(x, y, z):
    theta = (x - 304)*4/np.pi
    phi = (88 - y)*4/np.pi
    return theta, phi


class DataReader(object):
    def __init__(self, path, image_shape, lower_left, upper_right, divisions, num_classes):
        self._image_shape = image_shape
        self._lower_left = lower_left
        self._upper_right = upper_right
        self._divisions = divisions
        self._num_classes = num_classes
        self._path = os.path.abspath(path)
        self._image_data = self.get_filenames(path + '/image_data/training/')
        self._image_labels = self.get_filenames(path + '/image_labels/training/')
        self._velodyne_data = self.get_filenames(path + '/velodyne_data/training/')
        self._velodyne_labels = self.get_filenames(path + '/velodyne_labels/training/')

    def get_filenames(self, path):
        data_paths = os.listdir(path)
        data_paths = sorted(data_paths)
        data_paths = [path + data_path for data_path in data_paths]
        filenames = []
        for data_path in data_paths:
            _filenames = os.listdir(data_path)
            _filenames = sorted(_filenames)
            _filenames = [data_path + '/' + filename for filename in _filenames]
            filenames += _filenames
        return filenames

    def get_image_data(self):
        h, w, c = self._image_shape
        shape = (len(self._image_labels), h // 2, w // 2, c)
        img_data_loc = 'processed/img_data.npy'
        if os.path.exists(img_data_loc):
            image_data = np.load(img_data_loc)
            return image_data
        image_data = np.empty(shape)
        k = 0
        for filename in self._image_data:
            image = normalize_img(misc.imread(filename))  # Fix brightness and convert to lab colorspace
            image_data[k, :, :, :] = image[0:h:2, 0:w:2, 0:c]
            k += 1
        Path(os.path.dirname(img_data_loc)).mkdir(exist_ok=True)
        np.save(img_data_loc, image_data)
        return image_data

    def get_image_labels(self):
        h, w, _ = self._image_shape
        shape = (len(self._image_labels), h // 2, w // 2)
        img_labels_loc = 'processed/img_labels.npy'
        if os.path.exists(img_labels_loc):
            label_data = np.load(img_labels_loc)
            return label_data
        label_data = np.empty(shape)
        k = 0
        for filename in self._image_labels:
            label = io.loadmat(filename)
            label = label['truth']
            label_data[k, :, :] = label[0:h:2, 0:w:2]
            k += 1
        Path(os.path.dirname(img_labels_loc)).mkdir(exist_ok=True)
        np.save(img_labels_loc, label_data)
        return label_data

    def get_velodyne_data(self):
        shape = np.append(self._divisions, 1)
        shape = np.insert(shape, 0, len(self._velodyne_data))
        vel_data_loc = 'processed/vel_data.npy'
        if os.path.exists(vel_data_loc):
            velo_data = np.load(vel_data_loc)
            return velo_data
        velo_data = np.empty(shape)
        k = 0
        for filename in self._velodyne_data[:1]:
            velo = np.fromfile(filename, dtype='float32')
            velo = np.reshape(velo, [-1, 4])
            #velo = np.reshape(velo, [4, -1])
            #velo = np.transpose(velo)
            velo = velo[:, 0:3]
            velo = gen_occupancy_grid(velo, self._lower_left, self._upper_right, self._divisions)
            velo_data[k, :, :, :, :] = velo
            k += 1
            print(k)
        Path(os.path.dirname(vel_data_loc)).mkdir(exist_ok=True)
        np.save(vel_data_loc, velo_data)
        return velo_data

    def get_velodyne_labels(self):
        shape = np.insert(self._divisions, 0, len(self._velodyne_data))
        vel_labels_loc = 'processed/vel_labels.npy'
        if os.path.exists(vel_labels_loc):
            label_data = np.load(vel_labels_loc)
            return label_data
        label_data = np.empty(shape)
        k = 0
        for (data_filename, label_filename) in zip(self._velodyne_data, self._velodyne_labels):
            velo = np.fromfile(data_filename, dtype='float32')
            velo = np.reshape(velo, [-1, 4])
            #velo = np.reshape(velo, [4, -1])
            #velo = np.transpose(velo)
            velo = velo[:, 0:3]
            label = io.loadmat(label_filename)
            label = label['truth']
            velo = np.concatenate([velo, label], 1)
            velo = gen_label_occupancy_grid(velo, self._lower_left, self._upper_right, self._divisions, self._num_classes)
            label_data[k, :, :, :] = velo
            k += 1
            print(k)
        Path(os.path.dirname(vel_labels_loc)).mkdir(exist_ok=True)
        np.save(vel_labels_loc, label_data)
        return label_data


def original_to_label(original):
    return {
        3: 1,  # road
        5: 2,  # sidewalk
        6: 3,  # car
        7: 4,  # pedestrian
        8: 5  # cyclist
    }.get(original, 0)  # unkown


def label_to_original(label):
    return {
        1: 3,  # road
        2: 5,  # sidewalk
        3: 6,  # car
        4: 7,  # pedestrian
        5: 8  # cyclist
    }.get(label, 0)


def get_color(original):  # function to map ints to RGB array
    return {
        1: [153, 0, 0],  # building
        2: [0, 51, 102],  # sky
        3: [160, 160, 160],  # road
        4: [0, 102, 0],  # vegetation
        5: [255, 228, 196],  # sidewalk
        6: [255, 200, 50],  # car
        7: [255, 153, 255],  # pedestrian
        8: [204, 153, 255],  # cyclist
        9: [130, 255, 255],  # signage
        10: [193, 120, 87],  # fence
    }.get(original, [0, 0, 0])  # Unknown


def normalize_img(img):
    hist = equalize_adapthist(img) * 255  # equalize_adapthist turns into floats from [0,1]
    hist = hist.astype(np.uint8, copy=False)
    lab = rgb2lab(hist)
    return lab


def index_to_real(coord, ll, ur, divisions):
    interval = (ur-ll)/divisions
    return coord*interval+ll


def real_to_index(coord, ll, ur, divisions):
    interval = (ur-ll)/divisions
    return np.floor_divide((coord - ll), interval)


def put_kernels_on_grid(kernel, pad=1):
    """Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    NOTE: Code written by GitHub user kukuruza:
    https://gist.github.com/kukuruza/03731dc494603ceab0c5

    Args:
        kernel: tensor of shape [Y, X, NumChannels, NumKernels]
        pad: number of black pixels around each filter (between them)
    Return:
        Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return i, n // i
    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
    print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x

# dr = DataReader('/home/vdd6/Desktop/gen_seg_data', (374, 1238, 3))
# res = dr.get_image_data()
# res = dr.get_image_labels()
