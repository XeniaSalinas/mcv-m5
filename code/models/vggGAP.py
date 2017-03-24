# Keras imports
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D

from keras.applications.vgg16 import VGG16

# Paper: https://arxiv.org/pdf/1409.1556.pdf

def build_vggGAP(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    base_model = VGG16(include_top=False, weights=weights,
                       input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    
    x = Convolution2D(1024, 3, 3, activation='relu', border_mode='valid', name='conv_CAM')(x)
    
    x = GlobalAveragePooling2D(name="GAP")(x)
    x = Dense(n_classes, name='dense')(x)
    predictions = Activation("softmax", name="softmax")(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model
