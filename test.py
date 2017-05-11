import numpy as np
import matplotlib.pyplot as plt
import sys
import atexit

from model import GenSeg
from util import DataReader, original_to_label, label_to_original, get_color, normalize_img
from scipy import misc, io
from skimage.color import lab2rgb


def main():
    number = int(sys.argv[1])
    name = 'saved/Long7-Lab.ckpt'
    if number is 2: test2(name)
    elif number is 3: test3(name)
    else: test1(name)


def test3(name):
    input_shape = [None, 176, 608, 3]
    num_classes = 6

    dr = DataReader('data/', (352, 1216, 3))
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
    num_classes = 6
    filenames = [
        'data/image_data/testing/0000/000000.png',
        'data/image_data/testing/0000/000040.png',
        'data/image_data/testing/0004/000000.png',
        'data/image_data/testing/0005/000020.png',
        'data/image_data/testing/0005/000240.png'
    ]
    shape = (len(filenames), 176, 608, 3)
    n, h, w, c = shape
    image_data = np.zeros((n, h, w, c), dtype=np.uint8)

    i = 0
    for f in filenames:
        image = normalize_img(misc.imread(f))  # Fix brightness and convert to lab colorspace
        image_data[i, :, :, :] = image[:h*2:2, :w*2:2, :]
        '''plt.figure()
        plt.imshow(image_data[i])
        plt.figure()
        plt.imshow(lab2rgb(normalize_img(image_data[i])))
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
        colored[i, x, y] = get_color(label_to_original(value))

    i = 0
    for img in colored:
        img = img.astype(np.uint8, copy=False)
        misc.imsave('%d.png' % i, img, 'png')
        '''plt.figure()
        plt.imshow(img)
        plt.show()'''
        i += 1


def test1(name):
    input_shape = [None, 176, 608, 3]
    num_classes = 6

    dr = DataReader('data/', (352, 1216, 3))
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
    # average = 0
    # for k in range(n):
    #     idxs = np.array([k])
    #     batch_data = x[idxs,:,:,:]
    #     batch_labels = y[idxs,:,:]
    #     print(batch_data.shape)
    #     print(batch_labels.shape)
    #     result = model.apply(x)
    #     result = np.argmax(result, axis=-1)
    #     result = np.equal(result, y)
    #     result = np.average(result.astype(dtype=np.float32))
    #     average += result
    # average /= n
    # print("Average: ", average)
    # model.save_model()


if __name__ == "__main__":
    main()


