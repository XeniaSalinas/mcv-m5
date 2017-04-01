#### Utilities for SegNet
#### Xian Lopez Alvarez
#### 31/03/2017

import tensorflow as tf

# The indices obtained after doing a max-pooling operation come flattend.
# A maximum value at position [b, y, x, c] would have, in argmax, the following
# index: ((b * height + y) * width + x) * channels + c.
# The following function has been adapted from:
# https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.pack(output_list)