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
EPOCHS = 10
SEED = 1337

WIDTH = 250
HEIGHT = 250
IMAGES = 13233
BATCH_SIZE = 500


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
    for i in range(0, IMAGES, BATCH_SIZE):
        minibatch = train_data[i:BATCH_SIZE]
        model.train(minibatch, EPOCHS)
    training_end()

    train_results = model.apply(train_data)
    show_dataset(train_results)


def show_dataset(dataset):
    plot = None
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
        correct_w = WIDTH // 16 * 16
        correct_h = HEIGHT // 16 * 16
        print(correct_h, correct_w)
        dataset = np.empty([IMAGES, correct_w, correct_h, 3], dtype=np.ubyte)
        i = 0
        for root, dirs, files in os.walk("lfw"):
            for file in files:
                img_data = imread(root + '/' + file)
                dataset[i] = img_data[:correct_w, :correct_h, :]
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
