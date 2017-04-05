#### Utilities for SegNet
#### Xian Lopez Alvarez
#### 31/03/2017

import tensorflow as tf
import numpy as np

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

def argmax2coord(argmax, shape):
    coord = np.zeros(4)
    coord[0] = argmax // (shape[1] * shape[2] * shape[3])
    coord[1] = (argmax - coord[0] * shape[1] * shape[2] * shape[3]) // (shape[2] * shape[3])
    coord[2] = (argmax - shape[2] * shape[3] * (coord[1] + shape[1] * coord[0])) // shape[3]
    coord[3] = argmax - shape[3] * (coord[2] + shape[2] * (coord[1] + shape[1] * coord[0]))
    return coord