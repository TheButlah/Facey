import numpy as np
from model import GenSeg
import atexit
from util import DataReader
from scipy import misc, io


def main():
    test2()


def test2():
    num_classes = 11
    filenames = ['data/image_data/testing/0000/000000.png']
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
    result = np.argmax(result, axis=-1).astype(np.float32)

    colored = np.empty(shape)

    def get_color(label):  # function to map ints to RGB array
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
        }.get(label, [255, 255, 255])  # Unknown

    for (i, x, y), value in np.ndenumerate(result):
        colored[i, x, y] = get_color(value)

    i = 0
    for img in colored:
        misc.imsave('%d.png' % i, img)
        i += 1


def test1():
    input_shape = [None, 176, 608, 3]
    num_classes = 11

    dr = DataReader('data/', (352, 1216, 3))
    x = dr.get_image_data()
    y = dr.get_image_labels()
    n, _, _, _ = x.shape
    batch_size = 30
    iterations = 2500

    model = GenSeg(input_shape=input_shape, num_classes=num_classes)
    atexit.register(model.save_model)  # In case of ctrl-C
    averages = []
    for iteration in range(iterations):
        idxs = np.random.permutation(n)[:batch_size]
        batch_data = x[idxs,:,:,:]
        batch_labels = y[idxs,:,:]
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

