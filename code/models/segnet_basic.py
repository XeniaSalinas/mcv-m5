# Keras imports
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import DePool2D, CropLayer2D, NdSoftmax
from keras.layers import UpSampling2D

from keras import backend as K
dim_ordering = K.image_dim_ordering()


def build_segnet_basic(img_shape=(3, None, None), nclasses=8, depths=[64, 64, 64, 64],\
                       filter_size=7, l2_reg=0.):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    padded1 = ZeroPadding2D(padding=(1, 1), name='pad1')(inputs)

    enc1 = downsampling_block_basic(padded1, depths[0], filter_size, l2(l2_reg))
    enc2 = downsampling_block_basic(enc1, depths[1], filter_size, l2(l2_reg))
    enc3 = downsampling_block_basic(enc2, depths[2], filter_size, l2(l2_reg))
    enc4 = downsampling_block_basic(enc3, depths[3], filter_size, l2(l2_reg))

	  # ##### decoding layers

    dec1 = upsampling_block_basic(enc4, depths[3], filter_size, enc4, l2(l2_reg))
    dec2 = upsampling_block_basic(dec1, depths[2], filter_size, enc3, l2(l2_reg))
    dec3 = upsampling_block_basic(dec2, depths[1], filter_size, enc2, l2(l2_reg))
    dec4 = upsampling_block_basic(dec3, depths[0], filter_size, enc1, l2(l2_reg))
	
    l1 = Convolution2D(nclasses, 1, 1, border_mode='valid')(dec4)
    score = CropLayer2D(inputs, name='score')(l1)
    # Softmax
    softmax_segnet = NdSoftmax()(score)

    # Complete model
    model = Model(input=inputs, output=softmax_segnet)

    return model



def downsampling_block_basic(inputs, n_filters, filter_size,
    W_regularizer=None):
    # This extra padding is used to prevent problems with different input
    # sizes. At the end the crop layer remove extra paddings
    pad = ZeroPadding2D(padding=(1, 1))(inputs)
    conv = Convolution2D(n_filters, filter_size, filter_size,
    border_mode='same', W_regularizer=W_regularizer)(pad)
    bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
    act = Activation('relu')(bn)
    maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
    return maxp

def upsampling_block_basic(inputs, n_filters, filter_size, unpool_layer=None,
    W_regularizer=None, use_unpool=True):
    if use_unpool:
        up = DePool2D(unpool_layer)(inputs)
        return up
    else:
        up = UpSampling2D()(inputs)
        conv = Convolution2D(n_filters, filter_size, filter_size,
        border_mode='same', W_regularizer=W_regularizer)(up)
        bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
        return bn

# Keras dim orders
def channel_idx():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3
