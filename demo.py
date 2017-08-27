import numpy as np
import matplotlib.pyplot as plt
import os.path
import atexit
import tensorflow as tf

from model import Facey
from scipy.misc import imread
from sklearn.model_selection import train_test_split
from time import time
from glob import glob

FILENAME = 'dataset.npz'
MODEL_PATH = 'saved/overnight-sat.ckpt'
WEIGHTS_PATH = 'vgg16_weights.npz'

EPOCHS = 10
SEED = 1337

WIDTH = 250
HEIGHT = 250
IMAGES = 13233
BATCH_SIZE = 10

IMG_DEMO_DELAY = 0.01

weights = None


def main():
    dataset = load_dataset()
    train_data, test_data = train_test_split(dataset, train_size=0.8)
    del dataset
    model = load_model((None,) + train_data.shape[1:])

    def apply_demo():
        print("Applying model to training data:")
        apply(train_data[:10], model)
        print("Applying model to test data:")
        apply(test_data[:10], model)

    def train_demo():
        print("Training model...")
        train(train_data, model)

    apply_demo()


def apply(dataset, model):
    results = model.apply(dataset)
    show_dataset(results)


def train(train_data, model):

    def training_end():
        model.save_model(MODEL_PATH)
        elapsed_time = time() - start_time
        print("Elapsed time: %d" % elapsed_time)

    atexit.register(training_end)

    start_time = time()
    for i in range(0, IMAGES, BATCH_SIZE):
        minibatch = train_data[i:i+BATCH_SIZE]
        loss = model.train(minibatch, EPOCHS, start_stop_info=False, progress_info=True)
        # print("Loss Value: %f" % loss)
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
        plt.pause(IMG_DEMO_DELAY)
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
        dataset = np.empty([IMAGES, correct_w, correct_h, 3], dtype=np.ubyte)
        i = 0
        should_break = False
        for root, dirs, files in os.walk("lfw"):
            if should_break: break
            for file in files:
                img_data = imread(root + '/' + file)
                dataset[i] = img_data[:correct_w, :correct_h, :]
                i += 1
                if i >= IMAGES:
                    should_break = True
                    break
        np.savez_compressed(FILENAME, dataset=dataset)
    return dataset


def load_model(data_shape):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Make it so that the program does not grab all GPUs' memory at start

    glob(MODEL_PATH + "*")
    if len(glob(MODEL_PATH + "*")) is 3:
        kevin = Facey(input_shape=data_shape, seed=SEED, weights=weights, load_model=MODEL_PATH, config=config)
    else:
        kevin = Facey(input_shape=data_shape, weights=weights, seed=SEED, config=config)
    return kevin


if __name__ == "__main__":
    main()
