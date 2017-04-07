#### Layers for SegNet.
#### Xian Lopez Alvarez
#### Start: 31/03/2017
#### Last modified: 05/04/2017

import tensorflow as tf
from keras.layers.core import Layer


# Max pooling layer, keeping the indexes, for later unpooling.
# Made with TensorFlow.
# Only works for GPU, due to tf.nn.max_pool_with_argmax.
class max_pool_with_argmax(Layer):
    def __init__(self, stride, **kwargs):
        self.stride = stride
        super(max_pool_with_argmax, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = []
        # TODO: there may be problems with odd input shapes
        print 'input_shape.__class__.__name__ = ' + input_shape.__class__.__name__
        print 'output_shape.__class__.__name__ = ' + output_shape.__class__.__name__
        print 'self.stride.__class__.__name__ = ' + self.stride.__class__.__name__
        output_shape.append(input_shape[0])
        output_shape.append(input_shape[1] // self.stride)
        output_shape.append(input_shape[2] // self.stride)
        output_shape.append(input_shape[3])
        # output_shape = [output_shape, output_shape]
        print 'len([output_shape, output_shape]) = ' + str(len([output_shape, output_shape]))
        return output_shape, output_shape

    def call(self, x, mask=None):
        output, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,self.stride,self.stride,1], \
                                        strides=[1,self.stride,self.stride,1], padding='SAME')
        return output, argmax

# Unpooling with indexes:
# Made with TensorFlow.
class max_unpool_with_indices(Layer):
    def __init__(self, stride, **kwargs):
        self.stride = stride
        super(max_unpool_with_indices, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # output_shape = input_shape
        # output_shape[1] = input_shape[1] * self.stride
        # output_shape[2] = input_shape[2] * self.stride
        return [[input_shape[0]*input_shape[1]] + list(input_shape[2:])]

    def call(self, x, argmax=None):
        # Output shape:
        nbatch   = tf.slice(x.shape, begin=tf.constant([0]), size=tf.constant([1]))
        ncol     = tf.slice(x.shape, begin=tf.constant([1]), size=tf.constant([1])) * self.stride
        nrow     = tf.slice(x.shape, begin=tf.constant([2]), size=tf.constant([1])) * self.stride
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
    
    
    
    
    