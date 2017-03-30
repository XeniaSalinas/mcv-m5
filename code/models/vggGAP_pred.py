# Keras imports
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D

from keras.applications.vgg16 import VGG16

import numpy as np

# Paper: https://arxiv.org/pdf/1409.1556.pdf

def build_vggGAP_pred(img_shape=(3, 224, 224), n_classes=1000):

    # Get base model
    base_model = VGG16(include_top=False, weights='imagenet',
                       input_tensor=None, input_shape=img_shape)
    
    # Build the training model:
    x = base_model.get_layer('block5_conv3').output
    # Add one more convolution: # TODO: make sure the weights of this layer are loaded.
    x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv_CAM')(x)
    base_model_extended = Model(input=base_model.input, output=x)
    y = GlobalAveragePooling2D(name="GAP")(x)
    dense_layer = Dense(n_classes, name='dense', bias=False)
    y = dense_layer(y)
    predictions = Activation("softmax", name="softmax")(y)
    model_train = Model(input=base_model.input, output=predictions)
    
    # Get the weights of the new dense layer:
    model_train.load_weights('/home/master/m5_project/mcv-m5/code/weights/weights_tt100k_weak_vggGAP_short.hdf5')
#    model_train.load_weights('/home/xianlopez/Documents/weights_tt100k_weak_vggGAP_short.hdf5')
    weights_dense = dense_layer.get_weights()
    weights_forcam = weights_dense[0].reshape((1, 1, 1024, n_classes))
#    bias_forcam = np.zeros((n_classes,), dtype = np.float32)
    bias_forcam = weights_dense[1].reshape((n_classes))
    list_forcam = [weights_forcam, bias_forcam]
    
    # Layer to generate the Class Activation Map:
    layerCAM = Convolution2D(n_classes, 1, 1, border_mode='same', name='CAM')
    
    # New model, for predicting:
    x = base_model_extended.output
    predictions = layerCAM(x)

    # This is the model we will use
    model = Model(input=base_model.input, output=predictions)
    
    # Set the weights to the new layer:
    layerCAM.set_weights(list_forcam)



    return model
