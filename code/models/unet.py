#Unet based on https://gist.github.com/galtay/4565f0c100adca913fe2570f821e4331
# Keras imports
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers.core import Dropout
from keras.regularizers import l2

# Custom layers import
from layers.ourlayers import DePool2D, CropLayer2D, NdSoftmax
from layers.deconv import Deconvolution2D

from keras import backend as K
dim_ordering = K.image_dim_ordering()


def build_unet(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               init='glorot_uniform',
               freeze_layers_from=None, path_weights=None):

    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    padded1 = ZeroPadding2D(padding=(100, 100), name='padded1')(inputs)

    # Block 1
	conv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid', name='conv1_1', W_regularizer=l2(l2_reg))(input_layer)
    conv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid', name='conv1_2', W_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2,2), name='pool1')(conv1_2)

	# Block 2
    conv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid', name='conv2_1', W_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid', name='conv2_2', W_regularizer=l2(l2_reg))(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2,2), name='pool2')(conv2_2)
	
	# Block 3
    conv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid', name='conv3_1', W_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid', name='conv3_2', W_regularizer=l2(l2_reg))(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2,2), name='pool3')(conv3_2)

	# Block 4
    conv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid', name='conv4_1', W_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid', name='conv4_2', W_regularizer=l2(l2_reg))(conv4_1)
    conv4_drop = Dropout(0.5, name='conv4_drop')(conv4_2)
    pool4 = MaxPooling2D(pool_size=(2,2), name='pool4')(conv4_drop)

	# Block 5
    bottom_conv1 = Convolution2D(1024, 3, 3, init, 'relu', border_mode='valid', name='bottom_conv1', W_regularizer=l2(l2_reg))(pool4)
    bottom_conv2 = Convolution2D(1024, 3, 3, init, 'relu', border_mode='valid', name='bottom_conv2', W_regularizer=l2(l2_reg))(bottom_conv1)
    bottom_drop = Dropout(0.5, name='bottom_drop')(bottom_conv2)

	#uncoder blocks
	
	# Block 6
	deconv4 = Deconvolution2D(512, 2, 2, bottom_drop._keras_shape, init,'linear', border_mode='valid', subsample=(2, 2), name='deconv4', W_regularizer=l2(l2_reg))(bottom_drop)
    deconv4_crop = CropLayer2D(deconv4, name='deconv4_crop')(conv4_drop)
    deconv4_concat = merge([deconv4_crop, deconv4], mode='concat', concat_axis=3, name='deconv4_concat')
    deconv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid', name='deconv4_1', W_regularizer=l2(l2_reg))(deconv4_concat)
    deconv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='valid', name='deconv4_2', W_regularizer=l2(l2_reg))(deconv4_1)

	# Block 7
    deconv3 = Deconvolution2D(256, 2, 2,deconv4_2._keras_shape,init, 'linear', border_mode='valid', subsample=(2, 2), name='deconv3', W_regularizer=l2(l2_reg))(deconv4_2)
    deconv3_crop = CropLayer2D(deconv3, name='deconv3_crop')(conv3_2)
    deconv3_concat = merge([deconv3_crop, deconv3], mode='concat', concat_axis=3, name='deconv3_concat')
    deconv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid', name='deconv3_1', W_regularizer=l2(l2_reg))(deconv3_concat)
    deconv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='valid', name='deconv3_2', W_regularizer=l2(l2_reg))(deconv3_1)

	# Block 8
    deconv2 = Deconvolution2D(128, 2, 2, deconv3_2._keras_shape,init, 'linear', border_mode='valid', subsample=(2, 2), name='deconv2', W_regularizer=l2(l2_reg))(deconv3_2)
    deconv2_crop = CropLayer2D(deconv2, name='deconv2_crop')(conv2_2)
    deconv2_concat = merge([deconv2_crop, deconv2], mode='concat', concat_axis=3, name='deconv2_concat')
    deconv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid', name='deconv2_1', W_regularizer=l2(l2_reg))(deconv2_concat)
    deconv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='valid', name='deconv2_2', W_regularizer=l2(l2_reg))(deconv2_1)

	# Block 9
    deconv1 = Deconvolution2D(64, 2, 2, deconv2_2._keras_shape,init,'linear', border_mode='valid', subsample=(2, 2), name='deconv1', W_regularizer=l2(l2_reg))(deconv2_2)
    deconv1_crop = CropLayer2D(deconv1, name='deconv1_crop')(conv1_2)
    deconv1_concat = merge([deconv1_crop, deconv1], mode='concat', concat_axis=3, name='deconv1_concat')
    deconv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid', name='deconv1_1', W_regularizer=l2(l2_reg))(deconv1_concat)
    deconv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='valid', name='deconv1_2', W_regularizer=l2(l2_reg))(deconv1_1)

    l1 = Convolution2D(nclasses, 1, 1, border_mode='valid',name='logits')(deconv1_2)
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
		
def segnet_conv2D(inputs, nfilters, filter_rows, filter_cols, init, block, layer, l2_reg):
    name = 'conv' + block + '_' + layer
    x = Convolution2D(nfilters, filter_rows, filter_cols, init, border_mode='same',name=name, W_regularizer=l2(l2_reg))(inputs)
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
