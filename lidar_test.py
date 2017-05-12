import atexit
import numpy as np
import sys
import matplotlib.pyplot as plt


from model import GenSeg
from util import DataReader, original_to_label
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D

num_classes = 6
input_shape = [None, 96, 96, 96, 1]
datareader_params = ('data/', (352, 1216, 3), np.array([0, -32, -16]), np.array([64, 32, 16]), np.array([96, 96, 96]), 11)


def main():
    name = 'saved/lidar.ckpt'
    number = int(sys.argv[1])
    if number is 2: test2(name)
    elif number is 3: test3(name)
    else: test1(name)


def test3(name):
    dr = DataReader(*datareader_params)
    x = dr.get_velodyne_data()
    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model=name)
    datapoint = model.apply(x[:1, :, :, :, :])
    datapoint = np.argwhere(datapoint==3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(datapoint[:, 0], datapoint[:, 1], datapoint[:, 2])
    plt.show()


def test2(name):
    dr = DataReader(*datareader_params)
    x = dr.get_velodyne_data()
    y = dr.get_velodyne_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    datapoint = y[0, :, :, :]
    datapoint = np.argwhere(datapoint==3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(datapoint[:, 0], datapoint[:, 1], datapoint[:, 2])
    plt.show()


def test1(name):
    dr = DataReader(*datareader_params)
    x = dr.get_velodyne_data()
    y = dr.get_velodyne_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    n, _, _, _, _ = x.shape
    batch_size = 1
    iterations = sys.maxsize

    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model=name)
    atexit.register(model.save_model, name)  # In case of ctrl-C
    for iteration in range(iterations):
        idxs = np.random.permutation(n)[:batch_size]
        batch_data = x[idxs, :, :, :, :]
        batch_labels = y[idxs, :, :, :]
        print(iteration, model.train(
            x_train=batch_data, y_train=batch_labels,
            num_epochs=1, start_stop_info=False, progress_info=False
        ))


if __name__ == "__main__":
    main()


