import numpy as np
import matplotlib.pyplot as plt
import os.path
import atexit

from model import Facey
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from time import time

FILENAME = 'dataset.npz'
MODEL_PATH = 'saved/saved.ckpt'
EPOCHS = 1
SEED = 1337


def main():
    dataset = load_dataset()
    train_data, test_data = train_test_split(dataset, train_size=0.2)

    model = load_model((None,) + dataset.shape[1:])

    def training_end():
        model.save_model(MODEL_PATH)
        elapsed_time = start_time-time()
        print("Elapsed time: %d" % elapsed_time)

    atexit.register(training_end, model)

    start_time = time()
    model.train(train_data, EPOCHS)
    training_end()

    train_results = model.apply(train_data)
    show_dataset(train_results)


def show_dataset(dataset):
    plot = None
    print(dataset.shape)
    for img in dataset:
        if plot is None:
            plot = plt.imshow(img, vmin=0, vmax=255)
        else:
            plot.set_data(img)
        plt.pause(0.01)
        plt.draw()
    plt.close()


def load_dataset():
    if os.path.isfile(FILENAME):
        with np.load(FILENAME) as file:
            dataset = file['dataset']
    else:
        dataset = np.empty([13233, 250, 250, 3], dtype=np.ubyte)
        i = 0
        for root, dirs, files in os.walk("lfw"):
            for file in files:
                img_data = imread(root + '/' + file)
                dataset[i] = img_data
                i += 1
        np.savez_compressed(FILENAME, dataset=dataset)
    return dataset


def load_model(data_shape):
    if os.path.isfile(MODEL_PATH):
        kevin = Facey(input_shape=data_shape, seed=SEED, load_model=MODEL_PATH)
    else:
        kevin = Facey(input_shape=data_shape, seed=SEED)
    return kevin


if __name__ == "__main__":
    main()
