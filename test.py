import numpy as np
from model import GenSeg
import atexit


def main():
    shape = [10, 32, 32, 32, 1]
    input_shape = list(shape)
    input_shape[0] = None
    print(input_shape)
    num_classes = 7
    X = np.random.rand(*shape)  # allows the list to be passed as the various arguments to the function
    Y = np.random.random_integers(0, num_classes-1, size=shape[:-1])
    model = GenSeg(input_shape=input_shape, num_classes=num_classes)
    model.train(x_train=X, y_train=Y, num_epochs=50)
    result = model.apply(X)
    print("Probabilities: ", result.shape)
    result = np.argmax(result, axis=-1)
    print("Argmax: ", result.shape)
    result = np.equal(result, Y)
    print("Equality: ", result.shape)
    result = np.average(result.astype(dtype=np.float32))
    print("Average: ", result)


if __name__ == "__main__":
    main()
