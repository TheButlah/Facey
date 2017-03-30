import numpy as np
import tensorflow as tf
import segnet
import util


def main():
    # Create data
    x1 = np.random.rand(100, 1000, 3)
    x2 = np.random.rand(100, 1000, 3)
    y1 = np.random.randint(0, 2, (100, 1000, 2))
    y2 = np.random.randint(0, 2, (100, 1000, 2))

    X = np.array([x1, x2])
    Y = np.array([y1, y2])

    # Create network
    f = 3  # filter size
    k = 5  # num features
    N = 2  # num layers
    num_classes = 2

    x = tf.placeholder(tf.float32, [None, 100, 1000, 3])
    y = tf.placeholder(tf.float32, [None, 100, 1000, num_classes])
    W = tf.Variable(tf.random_normal([100, 1000, k, num_classes]))
    b = tf.Variable(tf.random_normal([100, 1000, num_classes]))

    h, masks = segnet.encode(x, f, k, N)
    y_hat = segnet.decode(h, masks, f, k)
    p_hat = tf.nn.softmax(util.tensor_mul(y_hat, W) + b)

    L = tf.reduce_mean(tf.squared_difference(y, p_hat))
    alpha = 0.01
    train = tf.train.GradientDescentOptimizer(alpha).minimize(L)

    # Test
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_epochs = 100
        for i in range(num_epochs):
            _, L_obs = sess.run([train, L], feed_dict={x: X, y: Y})
            print(L_obs)

if __name__ == "__main__":
    main()
