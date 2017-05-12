import numpy as np
import matplotlib.pyplot as plt
import sys
import atexit
import cv2

from model import GenSeg
from util import DataReader, original_to_label, label_to_original, get_color, normalize_img
from scipy.misc import imread, imsave
from pylab import rcParams

num_classes = 6
datareader_params = ('data/', (352, 1216, 3), np.array([0, -32, -16]), np.array([64, 32, 16]), np.array([64, 64, 64]), num_classes)


def main():
    number = int(sys.argv[1])
    name = 'saved/Long7-Lab-Fixed.ckpt'
    if number is 2: test2(name)
    elif number is 3: test3(name)
    elif number is 4: test4(name)
    else: test1(name)


def test4(name):
    rcParams['figure.figsize'] = 40, 15
    input_shape = [None, 176, 608, 3]

    dr = DataReader(*datareader_params)
    x = dr.get_image_data()
    y = dr.get_image_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    n, _, _, _ = x.shape
    batch_size = 30

    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model=name)

    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #out = cv2.VideoWriter('out.avi', fourcc, 20.0, tuple(input_shape[1:-1]))

    fig = None
    for i in range(0, n, batch_size):
        batch_data = x[i:i + batch_size, :, :, :]
        batch_labels = y[i:i + batch_size, :, :]
        results = model.apply(batch_data)
        results = np.argmax(results, axis=-1)
        for img in results:
            colored = np.empty(input_shape[1:], dtype=np.uint8)
            for (idx, idy), value in np.ndenumerate(img):
                colored[idx, idy, :] = get_color(label_to_original(value))
            if fig is None:
                fig = plt.imshow(colored)
            else:
                fig.set_data(colored)
            #plt.pause(.1)
            plt.draw()
            #out.write(colored.transpose().astype(np.uint8))

    #out.release()


def test3(name):
    input_shape = [None, 176, 608, 3]

    dr = DataReader(*datareader_params)
    x = dr.get_image_data()
    y = dr.get_image_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    n, _, _, _ = x.shape
    batch_size = 30

    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model=name)

    average = 0
    count = 0
    for i in range(0,n,batch_size):
        batch_data = x[i:i+batch_size, :, :, :]
        batch_labels = y[i:i+batch_size, :, :]
        results = model.apply(batch_data)
        results = np.argmax(results, axis=-1)
        results = np.equal(results, batch_labels)
        results = np.average(results.astype(dtype=np.float32))
        average += results
        count += 1
    print(average/count)


def test2(name):
    filenames = [
        'data/image_data/testing/0000/000000.png',
        'data/image_data/testing/0000/000040.png',
        'data/image_data/testing/0004/000000.png',
        'data/image_data/testing/0005/000020.png',
        'data/image_data/testing/0005/000240.png'
    ]
    shape = (len(filenames), 176, 608, 3)
    n, h, w, c = shape
    image_data = np.zeros((n, h, w, c))

    i = 0
    for f in filenames:
        image = normalize_img(imread(f))  # Fix brightness and convert to lab colorspace
        image_data[i, :, :, :] = image[:h*2:2, :w*2:2, :]
        '''plt.figure()
        plt.imshow(image_data[i])
        plt.figure()
        plt.imshow(lab2rgb(image_data[i]))
        plt.show()'''
        i += 1

    model = GenSeg(input_shape=[None, h, w, c], num_classes=num_classes, load_model=name)
    result = model.apply(image_data)
    result = np.argmax(result, axis=-1)

    '''for img in result:
        plt.figure()
        plt.imshow(img.astype(np.uint8))
    plt.show()'''

    colored = np.empty(shape)

    for (i, x, y), value in np.ndenumerate(result):
        colored[i, x, y] = get_color(list(label_to_original(value)))

    i = 0
    for img in colored:
        img = img.astype(np.uint8, copy=False)
        imsave('%d.png' % i, img, 'png')
        '''plt.figure()
        plt.imshow(img)
        plt.show()'''
        i += 1


def test1(name):
    input_shape = [None, 176, 608, 3]

    dr = DataReader(*datareader_params)
    x = dr.get_image_data()
    y = dr.get_image_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    n, _, _, _ = x.shape
    batch_size = 30
    iterations = sys.maxsize

    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model=name)
    atexit.register(model.save_model, name)  # In case of ctrl-C
    for iteration in range(iterations):
        idxs = np.random.permutation(n)[:batch_size]
        batch_data = x[idxs, :, :, :]
        batch_labels = y[idxs, :, :]
        print(iteration, model.train(
            x_train=batch_data, y_train=batch_labels,
            num_epochs=1, start_stop_info=False, progress_info=False
        ))


if __name__ == "__main__":
    main()


