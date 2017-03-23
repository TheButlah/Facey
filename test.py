import numpy as np
import tensorflow as tf
import segnet


def main():
    # Create data
    x1 = np.zeros((16, 16, 16, 1))
    x2 = np.zeros((16, 16, 16, 1))
    for i in range(16):
        for j in range(16):
            for k in range(16):
                n = i * 16*16 + j * 16 + k
                x1[i,j,k,:] = n #np.arange(3 * n, 3 * (n+1))
                x2[i,j,k,:] = 48 + n #np.arange(48 + 3 * n, 48 + 3 * (n+1))
    X = np.array([x1, x2])
    # Create network
    f = 2  # size of filter
    k = 5  # number of features
    N = 2  # number of layers

    x = tf.placeholder(tf.float32, [None, 16, 16, 16, 1])
    h, masks = segnet.encode(x, f, k, N)
    y_hat = segnet.decode(h, masks, f, k, 2)
    init = tf.global_variables_initializer()

    # Test
    with tf.Session() as sess:
        sess.run(init)
        Y_hat = sess.run(y_hat, feed_dict={x: X})

if __name__ == "__main__":
    main()
