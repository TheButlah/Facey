import numpy as np
from model import GenSeg
import atexit
from util import DataReader, original_to_label, label_to_original, get_color
from scipy import misc, io


def main():
    test1()


def test2():
    num_classes = 11
    filenames = ['data/image_data/testing/0000/000040.png']
    shape = (len(filenames), 176, 608, 3)
    n, h, w, c = shape
    image_data = np.zeros((n, h, w, c))

    i = 0
    for f in filenames:
        image = misc.imread(f)
        image_data[i, :, :, :] = image[:h*2:2, :w*2:2, :]
        i += 1

    model = GenSeg(input_shape=[None, h, w, c], num_classes=num_classes, load_model='saved/Calios.ckpt')
    result = model.apply(image_data)
    result = np.argmax(result, axis=-1)

    colored = np.empty(shape)

    for (i, x, y), value in np.ndenumerate(result):
        colored[i, x, y] = get_color(label_to_original(value))

    i = 0
    for img in colored:
        print(np.average(img[..., :1]))
        misc.imsave('%d.png' % i, img.astype(np.uint8), 'png')
        i += 1


def test1():
    input_shape = [None, 176, 608, 3]
    num_classes = 6

    dr = DataReader('data/', (352, 1216, 3))
    x = dr.get_image_data()
    y = dr.get_image_labels()
    func = np.vectorize(original_to_label)
    y = func(y)
    n, _, _, _ = x.shape
    batch_size = 30
    iterations = 2500

    model = GenSeg(input_shape=input_shape, num_classes=num_classes)
    #atexit.register(model.save_model)  # In case of ctrl-C
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

