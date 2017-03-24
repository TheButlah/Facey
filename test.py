import numpy as np
import tensorflow as tf
import segnet
import util


def main():
    # Create data
    x1 = np.zeros((16, 16, 3))
    x2 = np.zeros((16, 16, 3))
    y1 = np.zeros((16, 16, 2))
    y2 = np.zeros((16, 16, 2))
    r = 3
    for i in range(16):
        for j in range(16):
            n = i * 16 + j * 1
            x1[i,j,:] = np.arange(3 * n, 3 * (n+1))
            x2[i,j,:] = np.arange(48 + 3 * n, 48 + 3 * (n+1))
            y1[i,j,:] = np.arange(2)
            y2[i,j,:] = np.arange(2)
    X = np.array([x1, x2])
    Y = np.array([y1, y2])

    # Create network
    f = 3
    k = 12
    N = 2
    num_classes = 2

    x = tf.placeholder(tf.float32, [None, 16, 16, 3])
    p = tf.placeholder(tf.float32, [None, 16, 16, num_classes])
    W = tf.Variable(tf.random_normal([16, 16, k, num_classes]))
    b = tf.Variable(tf.random_normal([16, 16, num_classes]))

    h, masks = segnet.encode(x, f, k, N)
    y_hat = segnet.decode(h, masks, f, k)
    p_hat = tf.nn.softmax(util.tensor_mul(y_hat, W) + b)
    init = tf.global_variables_initializer()

    L = tf.reduce_mean(tf.squared_difference(p, p_hat))
    alpha = 100
    train = tf.train.GradientDescentOptimizer(alpha).minimize(L)

    # Test
    with tf.Session() as sess:
        sess.run(init)
        num_epochs = 100
        for i in range(num_epochs):
            _, L_obs = sess.run([train, L], feed_dict={x: X, p: Y})
            print L_obs

if __name__ == "__main__":
    main()
