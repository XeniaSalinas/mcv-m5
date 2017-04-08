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


def build_segnet_basic(img_shape=(3, None, None), nclasses=8, l2_reg=0.,\
                       freeze_layers_from=None, path_weights=None,\
                       depths=[64, 64, 64, 64],filter_size=7):

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

    # Load pretrained Model
    if path_weights:
        load_matcovnet(model, n_classes=nclasses)

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

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

# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True

# Keras dim orders
def channel_idx():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3

# Load weights from matconvnet
def load_matcovnet(model, path_weights='weights/pascal-fcn8s-dag.mat',
                   n_classes=11):

    import scipy.io as sio
    import numpy as np

    print('   Loading pretrained model: ' + path_weights)
    # Depending the model has one name or other
    if 'tvg' in path_weights:
        str_filter = 'f'
        str_bias = 'b'
    else:
        str_filter = '_filter'
        str_bias = '_bias'

    # Open the .mat file in python
    W = sio.loadmat(path_weights)

    # Load the parameter values into the model
    num_params = W.get('params').shape[1]
    for i in range(num_params):
        # Get layer name from the saved model
        name = str(W.get('params')[0][i][0])[3:-2]

        # Get parameter value
        param_value = W.get('params')[0][i][1]

        # Load weights
        if name.endswith(str_filter):
            raw_name = name[:-len(str_filter)]

            # Skip final part
            if n_classes==21 or ('score' not in raw_name and \
               'upsample' not in raw_name and \
               'final' not in raw_name and \
               'probs' not in raw_name):

                print ('   Initializing weights of layer: ' + raw_name)
                print('    - Weights Loaded (FW x FH x FC x K): ' + str(param_value.shape))

                if dim_ordering == 'th':
                    # TH kernel shape: (depth, input_depth, rows, cols)
                    param_value = param_value.T
                    print('    - Weights Loaded (K x FC x FH x FW): ' + str(param_value.shape))
                else:
                    # TF kernel shape: (rows, cols, input_depth, depth)
                    param_value = param_value.transpose((1, 0, 2, 3))
                    print('    - Weights Loaded (FH x FW x FC x K): ' + str(param_value.shape))

                # Load current model weights
                w = model.get_layer(name=raw_name).get_weights()
                print('    - Weights model: ' + str(w[0].shape))
                if len(w)>1:
                    print('    - Bias model: ' + str(w[1].shape))

                print('    - Weights Loaded: ' + str(param_value.shape))
                w[0] = param_value
                model.get_layer(name=raw_name).set_weights(w)

        # Load bias terms
        if name.endswith(str_bias):
            raw_name = name[:-len(str_bias)]
            if n_classes==21 or ('score' not in raw_name and \
               'upsample' not in raw_name and \
               'final' not in raw_name and \
               'probs' not in raw_name):
                print ('   Initializing bias of layer: ' + raw_name)
                param_value = np.squeeze(param_value)
                w = model.get_layer(name=raw_name).get_weights()
                w[1] = param_value
                model.get_layer(name=raw_name).set_weights(w)
    return model

if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_segnet_basic(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()