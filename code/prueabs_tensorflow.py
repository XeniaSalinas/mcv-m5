#### pruebas tensorflow

import tensorflow as tf
import numpy as np



sess = tf.InteractiveSession()

a = tf.placeholder(tf.float32, shape=(5,2,2,3))

# Only works with GPU !!
x1, argmax = tf.nn.max_pool_with_argmax(a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

val_a = np.zeros([5,2,2,3])
for idx1 in range(5):
    for idx2 in range(2):
        for idx3 in range(2):
            for idx4 in range(3):
                val_a[idx1, idx2, idx3, idx4] = (np.random.sample() - 0.5) * 10

res1 = sess.run([a], feed_dict={a:val_a})
res2 = sess.run([x1], feed_dict={a:val_a})









