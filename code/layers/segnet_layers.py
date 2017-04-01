#### Layers for SegNet.
#### Xian Lopez Alvarez
#### 31/03/2017

from tools.segnet_utils import unravel_argmax
import tensorflow as tf


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
    
    
    
    
    
    
    