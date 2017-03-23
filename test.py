import numpy as np
import tensorflow as tf
import segnet


def main():
    # Create data
    x1 = np.zeros((16, 16, 3))
    x2 = np.zeros((16, 16, 3))
    r = 3
    for i in range(16):
        for j in range(16):
            n = i * 16 + j * 1
            x1[i,j,:] = np.arange(3 * n, 3 * (n+1))
            x2[i,j,:] = np.arange(48 + 3 * n, 48 + 3 * (n+1))
    X = np.array([x1, x2])

    # Create network
    f = 2
    k = 5
    N = 2

    x = tf.placeholder(tf.float32, [None, 16, 16, 3])
    h, masks = segnet.encode(x, f, k, N)
    y_hat = segnet.decode(h, masks, f, k)
    init = tf.global_variables_initializer()

    # Test
    with tf.Session() as sess:
        sess.run(init)
        Y_hat = sess.run(y_hat, feed_dict={x: X})

if __name__ == "__main__":
    main()
