# Keras imports
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.regularizers import l2


# Paper: https://arxiv.org/pdf/1608.06993.pdf

def conv_block(input_tensor, filters, l2_reg, block, layer):
    """
    # Arguments
        input_tensor: input tensor
        filters: number of filters for the convolutional layers
        l2_reg: regularizer factor
        block: '1','2'..., current block label, used for generating layer names
        layer: '1','2'..., current layer label, used for generating layer names
    """
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'block' + str(block) + '_layer' + str(layer) + '_conv'
    bn_name_base = 'block' + str(block) + '_layer' + str(layer)  + '_bn'

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg))(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(4*filters, 1, 1, border_mode='same', name=conv_name_base + '1', W_regularizer=l2(l2_reg))(x)

    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2', gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    x = Convolution2D(filters, 3, 3, border_mode='same', name=conv_name_base + '2', W_regularizer=l2(l2_reg))(x)
    
    return x
    
def transition_block(input_tensor, filters, compression, l2_reg, block):
    """
    # Arguments
        input_tensor: input tensor
        filters: number of filters for the convolutional layers
        compression: output feature maps factor
        l2_reg: regularizer factor
        block: '1','2'..., current transition label, used for generating layer names
    """
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_name_base = 'trans' + str(block) + '_bn'
    conv_name_base = 'trans' + str(block) + '_conv'
    pool_name_base = 'trans' + str(block) + '_pool'

    x = BatchNormalization(axis=bn_axis, name=bn_name_base, gamma_regularizer=l2(l2_reg), beta_regularizer=l2(l2_reg))(input_tensor)
    x = Activation('relu')(x)
    x = Convolution2D(int(filters * compression), 1, 1, border_mode='same', name=conv_name_base, W_regularizer=l2(l2_reg))(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)
    
    return x
    
def dense_block(input_tensor, nb_layers, nb_filter, growth_rate, l2_reg, block):
    """
    # Arguments
        input_tensor: input_tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        l2_reg: regularizer factor
        block: '1','2'..., current transition label, used for generating layer names
    """

    if K.image_dim_ordering() == 'tf':
        concat_axis = -1
    else:
        concat_axis = 1

    feature_list = [input_tensor]
    x = input_tensor

    for layer in range(nb_layers):
        x = conv_block(x, growth_rate, l2_reg, block, layer)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter

def build_denseNet(img_shape=(3, 224, 224), n_classes=1000, depth=40, growth_rate=32, l2_reg=0.,
                load_pretrained=False, freeze_layers_from=None):
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None
    
    if K.image_dim_ordering() == 'tf':
        concat_axis = -1
    else:
        concat_axis = 1

    assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4'
    # layers in each dense block
    nb_layers = int((depth - 4) / 3)
        
    compression = 0.5
    nb_blocks = 3

    # compute initial nb_filter if -1, else accept users initial nb_filter
    nb_filter = 2 * growth_rate

    # Generate the model
    model_input = Input(shape=img_shape)

    # Initial block
    x = Convolution2D(nb_filter, 7, 7, border_mode='same', name='initial_conv', W_regularizer=l2(l2_reg))(model_input)

    # Add dense blocks
    for block in range(1,nb_blocks):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, l2_reg, block)
        # add transition_block
        x = transition_block(x, nb_filter, compression, l2_reg, block)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, l2_reg, nb_blocks)
    
    base_model = Model(input=model_input, output=x)
    
    # Classification block
    y = base_model.output
    y = GlobalAveragePooling2D()(y)
    predictions = Dense(n_classes, activation='softmax')(y)

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
