import numpy as np
from model import GenSeg
import atexit
from util import DataReader


def main():
    test1()

def test2():
    input_shape = [None, 176, 608, 3]
    num_classes = 11
    model = GenSeg(input_shape=input_shape, num_classes=num_classes, load_model='saved/Calios.ckpt'
    #Do something

def test1():
    input_shape = [None, 176, 608, 3]
    num_classes = 11

    dr = DataReader('/home/vdd6/Desktop/gen_seg_data', (352, 1216, 3))
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
