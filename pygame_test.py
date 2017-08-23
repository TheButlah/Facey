import pygame
import pygame.camera
import time
import numpy as np
import matplotlib.pyplot as plt
import os.path
import threading
from model import Facey
from scipy.misc import imread

FILENAME = 'dataset.npz'
MODEL_PATH = 'saved/saved.ckpt'
WEIGHTS_PATH = 'vgg16_weights.npz'
EPOCHS = 1
SEED = 1337

class Buffer:
    def __init__(self, intial_array):
        self.oldest_index = 0
        self.size = intial_array.shape[0]
        self.shape = intial_array.shape
        self.array = intial_array

    def add_image(self, image):
        self.array[self.oldest_index] = image
        self.oldest_index = (self.oldest_index + 1) % self.size

def show_dataset(dataset):
    plot = None
    # print(dataset.shape)
    for img in dataset:
        print(img)
        if plot is None:
            plot = plt.imshow(img, vmin=0, vmax=255)
        else:
            plot.set_data(img)
        plt.pause(0.0001)
        plt.draw()
    # plt.close()

def load_model(data_shape, weights):
    if os.path.isfile(MODEL_PATH):
        kevin = Facey(input_shape=data_shape, seed=SEED, weights=weights, load_model=MODEL_PATH)
    else:
        kevin = Facey(input_shape=data_shape, seed=SEED, weights=weights)
    return kevin

def load_dataset(buffer_size):
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
    return dataset[:buffer_size]

def stream_capture(buffer):
    # buffer = Buffer(np.zeros((200,480,640,3)))
    size = (640, 480)
    display = pygame.display.set_mode(size,0)
    pygame.camera.init()
    cam = pygame.camera.Camera(pygame.camera.list_cameras()[0], size)
    cam.start()
    snapshot = pygame.surface.Surface(size, 0, display)

    try:
        t0 = time.time()
        while True:
            if cam.query_image():
                snapshot = cam.get_image(snapshot)
                display.blit(snapshot, (0, 0))
                pygame.display.flip()
                if time.time() - t0 > 0.5:
                    img = cam.get_image()
                    raw = img.get_buffer().raw
                    arr = 255 - np.flip(np.frombuffer(raw, dtype=np.ubyte).reshape(size[1], size[0], 3), 2)
                    buffer.add_image(arr)
    except KeyboardInterrupt:
        pass

    cam.stop()
    show_dataset(buffer.array)
    # plt.imshow(buffer.array[-1])
    # plt.show()

def main():

    buffer_size = 3
    # initial_data = load_dataset(buffer_size)
    initial_data = np.zeros((buffer_size, 480, 640, 3))
    buffer = Buffer(initial_data)
    weights = np.load(WEIGHTS_PATH)
    kevin = load_model((None,) + buffer.shape[1:], weights)

    def train_model():
        while True:
            kevin.train(buffer.array, EPOCHS)

    def test_model():
        while True:
            show_dataset(kevin.apply(buffer.array))

    t1 = threading.Thread(target=stream_capture, args=(buffer,))
    # t2 = threading.Thread(target=train_model)
    t3 = threading.Thread(target=test_model)
    t1.start()
    # t2.start()
    t3.start()

if __name__ == '__main__':
    main()