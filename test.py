import numpy as np
import tensorflow as tf
import util

images = np.arange(96)
images = np.reshape(images, (-1, 4, 4, 3))
print "Image 1"
print images[0, :, :, 0]
print images[0, :, :, 1]
print images[0, :, :, 2]
print "Image 2"
print images[1, :, :, 0]
print images[1, :, :, 1]
print images[1, :, :, 2]

x = tf.placeholder(tf.float32, [None, 4, 4, 3])
y = util.down_sample(x, 2)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    y_obs = sess.run(y, feed_dict={x: images})
    print "Result 1"
    print y_obs[0, :, :, 0]
    print y_obs[0, :, :, 1]
    print y_obs[0, :, :, 2]
    print "Result 2"
    print y_obs[1, :, :, 0]
    print y_obs[1, :, :, 1]
    print y_obs[1, :, :, 2]
