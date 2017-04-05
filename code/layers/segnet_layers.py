#### Layers for SegNet.
#### Xian Lopez Alvarez
#### 31/03/2017

import tensorflow as tf
import numpy as np


# Max pooling layer, keeping the indexes, for later unpooling.
class max_pool_keep_idxs():
#    tf.nn.max_pool_with_argmax(
#        input,
#        ksize,
#        strides,
#        padding,
#        Targmax=None,
#        name=None
#    )

    # output, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    pass

# Unpooling:
def max_unpool(x, indexes, stride):
    # I assume (impose) zero-padding.
    # I assume the same stride for x and y (so stride is a scalar)

    # Compute output shape, from input shape and stride.
    # The output number of rows and columns will always be even, so there
    # can be problems if, before doing the max-pooling, we had an odd number,
    # because the output shape of this layer will not be the same to the
    # one of the layer we had before doing the pooling and unpooling
    # operations.
    in_shape = tf.shape(x)
    out_shape = in_shape
    out_shape[1] = in_shape[1] * stride
    out_shape[2] = in_shape[2] * stride
    
    # Initialize output:
    output = tf.zeros(out_shape)

# Unpooling with indexes:
def max_unpool_with_indices(x, argmax, stride=2):

    # Output shape:
    nbatch   = tf.slice(x.shape, begin=tf.constant([0]), size=tf.constant([1]))
    ncol     = tf.slice(x.shape, begin=tf.constant([1]), size=tf.constant([1])) * stride
    nrow     = tf.slice(x.shape, begin=tf.constant([2]), size=tf.constant([1])) * stride
    nchannel = tf.slice(x.shape, begin=tf.constant([3]), size=tf.constant([1]))
    shape = tf.concat([nbatch, ncol, nrow, nchannel], axis=0)
    shape = tf.cast(shape, dtype=tf.int64)

    # Scalars that will be used:
    nbatch_scalar = tf.reshape(nbatch, [])
    nchannel_scalar = tf.reshape(nchannel, [])

    # Modify the argmax to include the batch:
    aux1 = tf.range(nbatch_scalar) * nrow * ncol * nchannel
    aux1 = tf.expand_dims(aux1, axis=1)
    aux1 = tf.expand_dims(aux1, axis=2)
    aux1 = tf.expand_dims(aux1, axis=3)
    aux2 = tf.constant([1, 1, 1], dtype=tf.int32)
    aux2 = tf.concat([aux2, nchannel], axis=0)
    aux3 = tf.tile(aux1, aux2)
    aux3 = tf.cast(aux3, dtype=tf.int64)
    argmax = argmax + aux3

    # Unravel the argmax:
    tcoord0 = argmax // (shape[1] * shape[2] * shape[3])
    tcoord1 = (argmax - tcoord0 * shape[1] * shape[2] * shape[3]) // (shape[2] * shape[3])
    tcoord2 = (argmax - shape[2] * shape[3] * (tcoord1 + shape[1] * tcoord0)) // shape[3]
    tcoord3 = argmax - shape[3] * (tcoord2 + shape[2] * (tcoord1 + shape[1] * tcoord0))

    # Stack the coordinates over a new dimension:
    tcoord = tf.stack([tcoord0, tcoord1, tcoord2, tcoord3], axis=4)

    # Create the output vector, which has zeros everywhere but where we have an index:
    output = tf.scatter_nd(tcoord, x, shape)

    return output
    
    
    
    
    