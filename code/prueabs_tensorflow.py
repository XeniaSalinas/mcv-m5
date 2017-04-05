#### pruebas tensorflow

import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, "/home/master/remote_xian/code")

from tools.segnet_utils import unravel_argmax
from tools.segnet_utils import argmax2coord
from layers.segnet_layers import max_unpool_with_indices


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
res3 = sess.run([argmax], feed_dict={a:val_a})

a1 = np.squeeze(np.array(res3), 0)
x1_val = np.squeeze(np.array(res2), 0)
matrix = np.squeeze(np.array(res1), 0)


print a1[0,0,0,0]
print x1_val[0,0,0,0]
print matrix[0,:,:,0]

argmax_unravel = unravel_argmax(argmax, matrix.shape)
res4 = sess.run([argmax_unravel], feed_dict={a:val_a})



###############################################



b = tf.placeholder(tf.float32, shape=(1,2,2,2))
x2, argmax2 = tf.nn.max_pool_with_argmax(b, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
val_b = np.zeros([1,2,2,2])
val_b[0,0,0,0] = -1.
val_b[0,0,1,0] = 0.
val_b[0,1,0,0] = -5.
val_b[0,1,1,0] = 2.2
val_b[0,0,0,1] = -1.
val_b[0,0,1,1] = 10.
val_b[0,1,0,1] = -5.
val_b[0,1,1,1] = 2.2
res12, res22, res32 = sess.run([b, x2, argmax2], feed_dict={b:val_b})
bval = np.squeeze(np.array(res12), 0)
x2_val = np.squeeze(np.array(res22), 0)
argmax2_val = np.squeeze(np.array(res32), 0)

# nsamples, height, width, nchannels = val_b.shape
#
# index = 3
#
# b = index // (height * width * nchannels)
# y = (index - b * height * width * nchannels) // (width * nchannels)
# x = (index - width * nchannels * (y + height * b)) // nchannels
# c = index - nchannels * (x + width * (y + height * b))
#
# b = tf.placeholder(tf.float32, shape=(1,2,2,2))
# insertion = tf.placeholder(tf.float32, shape=(1))
# inser_val = 100.
# c = b
# b[0,0,0,1] = insertion
# res4 = sess.run([c], feed_dict={b:val_b, insertion:inser_val})


indices = argmax2
update = x2
shape = tf.constant([1,2,2,2], dtype=tf.int64)

x3 = tf.scatter_nd(indices=indices, updates=update, shape=shape)


######################
indices = tf.placeholder(dtype=tf.int32, shape=(4,1))
updates = tf.placeholder(dtype=tf.int32, shape=(4))
indices_val = np.array([[4], [3], [1], [7]])
updates_val = np.array([9, 10, 11, 12])


# indices = tf.constant([[4], [3], [1], [7]])
# updates = tf.constant([9, 10, 11, 12])
shape = tf.constant([8])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})


######################
indices = tf.placeholder(dtype=tf.int32, shape=(6,1))
updates = tf.placeholder(dtype=tf.int32, shape=(6,2))
indices_val = np.array([[4], [3], [1], [7], [2], [11]])
updates_val = np.array([[9, 90], [10, 100], [11, 110], [12, 120], \
                            [19, 190], [212, 2120]])


shape = tf.constant([12,2])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})


######################
indices = tf.placeholder(dtype=tf.int32, shape=(2,1))
updates = tf.placeholder(dtype=tf.int32, shape=(2,2,3))
indices_val = np.array([[2], [1]])
updates_val = np.array([[[6,6,7],[69,68,70]], [[-50,-80,-70],[-5,-8,-7]]])


shape = tf.constant([3,2,3])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})
scatter_val


######################
indices = tf.placeholder(dtype=tf.int32, shape=(2,1))
updates = tf.placeholder(dtype=tf.int32, shape=(2,2,5))
indices_val = np.array([[1], [3]])
updates_val = np.array([[[6,7,8,9,10],[69,70,71,72,73]], [[-50,51,52,53,54],[-5,-6,-7,-8,-9]]])


shape = tf.constant([4,2,5])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})
scatter_val


######################
indices = tf.placeholder(dtype=tf.int32, shape=(3,4))
updates = tf.placeholder(dtype=tf.int32, shape=(3))
indices_val = np.array([[0,0,1,0], [0,1,1,0], [0,3,2,0]])
updates_val = np.array([6, 69, -7])

shape = tf.constant([1,4,4,4])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})
scatter_val


######################
indices = tf.placeholder(dtype=tf.int32, shape=(2,3,4))
updates = tf.placeholder(dtype=tf.int32, shape=(2,3))
indices_val = np.array([[[0,0,1,0], [0,1,1,0], [0,3,2,0]], [[0,0,1,1], [0,1,1,2], [0,3,2,3]]])
updates_val = np.array([[6, 69, -7], [1, 2, 3]])

shape = tf.constant([1,4,4,4])
scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={indices:indices_val, updates:updates_val})
scatter_val


######################

shape = [1,2,2,2]
argmax2coord(6, shape)
argmax2coord(3, shape)


######################
a = tf.placeholder(tf.float32, shape=(5,2,2,3))
# Only works with GPU !!
x1, argmax = tf.nn.max_pool_with_argmax(a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', Targmax=tf.int64)
val_a = np.zeros([5,2,2,3])
for idx1 in range(5):
    for idx2 in range(2):
        for idx3 in range(2):
            for idx4 in range(3):
                val_a[idx1, idx2, idx3, idx4] = (np.random.sample() - 0.5) * 10
a_val, x1_val, argmax_val = sess.run([a, x1, argmax], feed_dict={a:val_a})

shape = tf.constant([5,2,2,3], dtype=tf.int64)
shape_val = [5,2,2,3]

indices = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3,4))
updates = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3))


# mod_argmax_1 = np.ones([1,1,3])

# Replicar este vector, añadiendo dimensiones, etc., y usarlo para modificar argmaxs

# The
nbatch   = tf.slice(x1.shape, begin=tf.constant([0]), size=tf.constant([1]))
ncol     = tf.slice(x1.shape, begin=tf.constant([1]), size=tf.constant([1])) * 2
nrow     = tf.slice(x1.shape, begin=tf.constant([2]), size=tf.constant([1])) * 2
nchannel = tf.slice(x1.shape, begin=tf.constant([3]), size=tf.constant([1]))
shape = tf.concat([nbatch, ncol, nrow, nchannel], axis=0)

# Scalars that will be used:
nbatch_scalar = tf.reshape(nbatch, [])
nchannel_scalar = tf.reshape(nchannel, [])

aux1 = tf.range(nbatch_scalar) * nrow * ncol * nchannel
aux1 = tf.expand_dims(aux1, axis=1)
aux1 = tf.expand_dims(aux1, axis=2)
aux1 = tf.expand_dims(aux1, axis=3)
aux2 = tf.constant([1, 1, 1], dtype=tf.int32)
aux2 = tf.concat([aux2, nchannel], axis=0)
aux3 = tf.tile(aux1, aux2)
aux3 = tf.cast(aux3, dtype=tf.int64)
argmax = argmax + aux3
eval = sess.run([aux3], feed_dict={a:val_a})
eval

mod_argmax_2 = np.zeros([5,1,1,1])
for idx in range(5):
    mod_argmax_2[idx,0,0,0] = idx * (shape_val[1] * shape_val[2] * shape_val[3])
t_mod_argmax = tf.constant(mod_argmax_2, dtype=tf.int64)
t_multiples = tf.constant([1,1,1,3], dtype=tf.int32)
t_mod_argmax_2 = tf.tile(t_mod_argmax, t_multiples)
# t_mod_argmax_2_val = sess.run([t_mod_argmax_2], feed_dict={a:val_a})
argmax = argmax + t_mod_argmax_2

# Initialize coordinate tensors:
tcoord0 = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3))
tcoord1 = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3))
tcoord2 = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3))
tcoord3 = tf.placeholder(dtype=tf.int64, shape=(5,1,1,3))

# Unravel the argmax:
tcoord0 = argmax // (shape[1] * shape[2] * shape[3])
tcoord1 = (argmax - tcoord0 * shape[1] * shape[2] * shape[3]) // (shape[2] * shape[3])
tcoord2 = (argmax - shape[2] * shape[3] * (tcoord1 + shape[1] * tcoord0)) // shape[3]
tcoord3 = argmax - shape[3] * (tcoord2 + shape[2] * (tcoord1 + shape[1] * tcoord0))

# Stack the coordinates over a new dimension:
tcoord = tf.stack([tcoord0, tcoord1, tcoord2, tcoord3], axis=4)

# # Add one dimension:
# tcoord0 = tf.expand_dims(tcoord0, axis=4)
# tcoord1 = tf.expand_dims(tcoord1, axis=4)
# tcoord2 = tf.expand_dims(tcoord2, axis=4)
# tcoord3 = tf.expand_dims(tcoord3, axis=4)
#
# # Concatenate along the new dimension:
# tcoord = tf.concat([tcoord0, tcoord1, tcoord2, tcoord3], axis=4)

tcoord_val = sess.run([tcoord], feed_dict={a:val_a})

indices = tcoord
updates = x1

scatter = tf.scatter_nd(indices, updates, shape)

scatter_val = sess.run([scatter], feed_dict={a:val_a})

x_unpooled = np.squeeze(np.array(scatter_val), 0)

x_unpooled[0,:,:,0]
a_val[0,:,:,0]

print ''

x_unpooled[1,:,:,2]
a_val[1,:,:,2]

print ''

x_unpooled[3,:,:,1]
a_val[3,:,:,1]


# parte = tf.slice(a, begin=tf.constant([3,0,0,1]), size=tf.constant([1,2,2,1]))
parte0 = tf.slice(a.shape, begin=tf.constant([0]), size=tf.constant([1]))
parte1 = tf.slice(a.shape, begin=tf.constant([1]), size=tf.constant([1]))
parte2 = tf.slice(a.shape, begin=tf.constant([2]), size=tf.constant([1]))
parte3 = tf.slice(a.shape, begin=tf.constant([3]), size=tf.constant([1]))
parte = tf.concat([parte0, parte1*2, parte2*2, parte3], axis=0)
parte_val = sess.run([parte], feed_dict={a:val_a})
parte_val


shape_eval = sess.run([shape], feed_dict={a:val_a})
shape_eval


############################################
############################################
############################################
a = tf.placeholder(tf.float32, shape=(5,2,2,3))

# Pool:
x1, argmax = tf.nn.max_pool_with_argmax(a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', Targmax=tf.int64)

# Unpool:
x2 = max_unpool_with_indices(x1, argmax, stride=2)

# Give values:
val_a = np.zeros([5,2,2,3])
for idx1 in range(5):
    for idx2 in range(2):
        for idx3 in range(2):
            for idx4 in range(3):
                val_a[idx1, idx2, idx3, idx4] = (np.random.sample() - 0.5) * 10

# Obtain results:
a_val, x1_val, x2_val = sess.run([a, x1, x2], feed_dict={a:val_a})
