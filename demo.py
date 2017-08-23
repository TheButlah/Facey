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
WEIGHTS_PATH = 'vgg16_weights.npz'

EPOCHS = 10
SEED = 1337

WIDTH = 250
HEIGHT = 250
IMAGES = 100
BATCH_SIZE = 1

weights = None

def main():
    dataset = load_dataset()
    train_data, test_data = train_test_split(dataset, train_size=0.8)
    #del dataset
    model = load_model((None,) + train_data.shape[1:])

    def training_end():
        model.save_model(MODEL_PATH)
        elapsed_time = time()
        print("Elapsed time: %d" % elapsed_time)

    atexit.register(training_end, model)

    start_time = time()
    for i in range(0, IMAGES, BATCH_SIZE):
        minibatch = train_data[i:BATCH_SIZE]
        loss = model.train(minibatch, EPOCHS, start_stop_info=False, progress_info=False)
        print("Loss Value: %f" % loss)
    training_end()

    train_results = model.apply(train_data[:10])
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
    global weights
    weights = np.load(WEIGHTS_PATH)
    if os.path.isfile(FILENAME):
        with np.load(FILENAME) as file:
            dataset = file['dataset']
    else:
        correct_w = WIDTH // 16 * 16
        correct_h = HEIGHT // 16 * 16
        print(correct_h, correct_w)
        dataset = np.empty([IMAGES, correct_w, correct_h, 3], dtype=np.ubyte)
        i = 0
        should_break = False
        for root, dirs, files in os.walk("lfw"):
            if should_break: break
            for file in files:
                img_data = imread(root + '/' + file)
                dataset[i] = img_data[:correct_w, :correct_h, :]
                i += 1
                if i > BATCH_SIZE:
                    should_break = True
                    break 
        np.savez_compressed(FILENAME, dataset=dataset)
    assert np.all(np.isfinite(dataset))
    return dataset


def load_model(data_shape):
    if os.path.isfile(MODEL_PATH):
        kevin = Facey(input_shape=data_shape, seed=SEED, weights=weights, load_model=MODEL_PATH)
    else:
        kevin = Facey(input_shape=data_shape, weights=weights, seed=SEED)
    return kevin


if __name__ == "__main__":
    main()
