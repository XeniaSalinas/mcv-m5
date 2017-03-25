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
#    x = base_model.output
    x = base_model.get_layer('block5_conv3').output
    x = Convolution2D(1024, 3, 3, activation='relu', border_mode='same', name='conv_CAM')(x)
    base_model_extended = Model(input=base_model.input, output=x)
    y = GlobalAveragePooling2D(name="GAP")(x)
    dense_layer = Dense(n_classes, name='dense')
    y = dense_layer(y)
    predictions = Activation("softmax", name="softmax")(y)
    model_train = Model(input=base_model.input, output=predictions)
    
    print 'nclasses = ', n_classes
    
    # Get the layes of the new dense layer:
#    model_train.load_weights('/home/xianlopez/Documents/myvenv1/tt100k_vggGAP/weights.hdf5')
    model_train.load_weights('/home/xianlopez/Documents/weights_tt100k_vggGAP_conv.hdf5')
    weights_dense = dense_layer.get_weights()
    print 'weights_dense.__class__.__name__ = ' + weights_dense.__class__.__name__
    print 'len(weights_dense) = ' + str(len(weights_dense))
    print 'weights_dense[0].shape = ' + str(weights_dense[0].shape)
    print 'weights_dense[0].dtype = ' + str(weights_dense[0].dtype)
    print 'weights_dense[1].shape = ' + str(weights_dense[1].shape)
    print 'weights_dense[1].dtype = ' + str(weights_dense[1].dtype)
    
#    base_model.summary()
#    weights = base_model.layers[2].get_weights()
#    print 'weights.__class__.__name__ = ' + weights.__class__.__name__
#    print 'len(weights) = ' + str(len(weights))
#    print 'weights[0].shape = ' + str(weights[0].shape)
#    print 'weights[1].shape = ' + str(weights[1].shape)

#    model_train.summary()
    base_model.summary()
    
#    weights_forcam = weights_dense[0].reshape((1, 1, 512, n_classes))
    weights_forcam = weights_dense[0].reshape((1, 1, 1024, n_classes))
    print 'weights_forcam.shape = ' + str(weights_forcam.shape)
    
    bias_forcam = np.zeros((n_classes,), dtype = np.float32)
    print 'bias_forcam.shape = ' + str(bias_forcam.shape)
    print 'bias_forcam.dtype = ' + str(bias_forcam.dtype)
    
    list_forcam = [weights_forcam, bias_forcam]
    
    # Layer to generate the Class Activation Map:
    layerCAM = Convolution2D(n_classes, 1, 1, border_mode='same', name='CAM')
    
    # New model, for predicting:
    x = base_model_extended.output
    predictions = layerCAM(x)

    # This is the model we will use
    model = Model(input=base_model.input, output=predictions)
    
    
    
    
    print 'input = ' + str(layerCAM.input)
    print 'output = ' + str(layerCAM.output)
    print 'input_shape = ' + str(layerCAM.input_shape)
    print 'output_shape = ' + str(layerCAM.output_shape)
    
#    weigths_layerCAM = layerCAM.get_weights()
    layerCAM.set_weights(list_forcam)
    weigths_layerCAM = layerCAM.get_weights()
    print 'weigths_layerCAM.__class__.__name__ = ' + weigths_layerCAM.__class__.__name__
    print 'len(weights_layerCAM) = ' + str(len(weigths_layerCAM))
    print 'weights_layerCAM[0].shape = ' + str(weigths_layerCAM[0].shape)
    print 'weights_layerCAM[1].shape = ' + str(weigths_layerCAM[1].shape)



    return model
