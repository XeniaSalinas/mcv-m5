# Keras imports
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import DePool2D, CropLayer2D, NdSoftmax
from layers.deconv import Deconvolution2D

from keras import backend as K
dim_ordering = K.image_dim_ordering()


def build_segnet_vgg(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               freeze_layers_from=None):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    padded1 = ZeroPadding2D(padding=(1, 1), name='pad1')(inputs)

    # Block 1
    conv1_1 = segnet_conv2D(padded1, 64, 3, 3, init, block='1',layer='1', l2_reg)
    conv1_2 = segnet_conv2D(conv1_1, 64, 3, 3, init, block='1',layer='2', l2_reg)
    pool1 = MaxPooling2D((2, 2), (2, 2), name='pool1')(conv1_2)

    # Block 2
    padded2 = ZeroPadding2D(padding=(1, 1), name='pad2')(pool1)
    conv2_1 = segnet_conv2D(padded2,128, 3, 3, init, block='2',layer='1', l2_reg)
    conv2_2 = segnet_conv2D(conv2_1, 128, 3, 3, init, block='2',layer='2', l2_reg)
    pool2 = MaxPooling2D((2, 2), (2, 2), name='pool2')(conv2_2)

    # Block 3
    padded3 = ZeroPadding2D(padding=(1, 1), name='pad3')(pool2)
    conv3_1 = segnet_conv2D(padded3, 256, 3, 3, init,  block='3',layer='1', l2_reg)
    conv3_2 = segnet_conv2D(conv3_1,256, 3, 3, init, block='3',layer='2', l2_reg)
    conv3_3 = segnet_conv2D(conv3_2,256, 3, 3, init, block='3',layer='3', l2_reg)
    pool3 = MaxPooling2D((2, 2), (2, 2), name='pool3')(conv3_3)

    # Block 4
    padded4 = ZeroPadding2D(padding=(1, 1), name='pad4')(pool3)
    conv4_1 = segnet_conv2D(padded4, 512, 3, 3, init, block='4',layer='1', l2_reg)
    conv4_2 = segnet_conv2D(conv4_1,512, 3, 3, init, block='4',layer='2', l2_reg)
    conv4_3 = segnet_conv2D(conv4_2,512, 3, 3, init, block='4',layer='3', l2_reg)
    pool4 = MaxPooling2D((2, 2), (2, 2), name='pool4')(conv4_3)

    # Block 5
    padded5 = ZeroPadding2D(padding=(1, 1), name='pad5')(pool4)
    conv5_1 = segnet_conv2D(padded5,512, 3, 3, init, block='5',layer='1', l2_reg)
    conv5_2 = segnet_conv2D(conv5_1,512, 3, 3, init, block='5',layer='2', l2_reg)
    conv5_3 = segnet_conv2D(conv5_2,512, 3, 3, init, block='5',layer='3', l2_reg)
    pool5 = MaxPooling2D((2, 2), (2, 2), name='pool5')(conv5_3)

	  # ##### decoding layers
	
	  # Block 6: Unpooling block 5
    unpool5 = DePool2D(pool2d_layer=pool5, size=(2,2), name='unpool_block5')(pool5)
    conv6_1 = segnet_conv2D(unpool5,512, 3, 3, init, block='6',layer='1', l2_reg)
    conv6_2 = segnet_conv2D(conv6_1,512, 3, 3, init, block='6',layer='2', l2_reg)
    conv6_3 = segnet_conv2D(conv6_2,512, 3, 3, init, block='6',layer='3', l2_reg)
	
	  # Block 7: Unpooling block 4
    unpool4 = DePool2D(pool2d_layer=pool4, size=(2,2), name='unpool_block4')(conv6_3)
    conv7_1 = segnet_conv2D(unpool4, 512, 3, 3, init, block='7',layer='1', l2_reg)
    conv7_2 = segnet_conv2D(conv7_1,512, 3, 3, init, block='7',layer='2', l2_reg)
    conv7_3 = segnet_conv2D(conv7_2,512, 3, 3, init, block='7',layer='3', l2_reg)
	
	  # Block 8: Unpooling block 3
    unpool3 = DePool2D(pool2d_layer=pool3, size=(2,2), name='unpool_block3')(conv7_3)
    conv8_1 = segnet_conv2D(unpool3, 256, 3, 3, init,  block='8',layer='1', l2_reg)
    conv8_2 = segnet_conv2D(conv8_1,256, 3, 3, init, block='8',layer='2', l2_reg)
    conv8_3 = segnet_conv2D(conv8_2,256, 3, 3, init, block='8',layer='3', l2_reg)

	  # Block 9: Unpooling block 2
    unpool2 = DePool2D(pool2d_layer=pool2, size=(2,2), name='unpool_block2')(conv8_3)
    conv9_1 = segnet_conv2D(unpool2,128, 3, 3, init, block='9',layer='1', l2_reg)
    conv9_2 = segnet_conv2D(conv9_1, 128, 3, 3, init, block='9',layer='2', l2_reg)
	
	  # Block 10: Unpooling block 1
    unpool1 = DePool2D(pool2d_layer=pool1, size=(2,2), name='unpool_block1')(conv9_2)
    conv10_1 = segnet_conv2D(unpool1,64, 3, 3, init, block='10',layer='1', l2_reg)
    conv10_2 = segnet_conv2D(conv10_1, 64, 3, 3, init, block='10',layer='2', l2_reg)
	
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
		
def segnet_conv2D(inputs, nfilters, filter_rows, filter_cols, init='glorot_uniform', block, layer, l2_reg=None):
    name = 'conv' + block + '_' + layer
    x = Convolution2D(n_filters, filter_rows, filter_cols, init, border_mode='same',name=name, W_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization(mode=0, axis=channel_idx(), name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    return x
						   
		

# Lad weights from matconvnet
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
    model = build_segnet_vgg(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
