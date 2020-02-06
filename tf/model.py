import keras
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Add

from keras.models import Model
from attention import *

def Mymodel(input_shape=32,
            classes=10,
            _isseblock=False,
            _iscbamblock=False,
            _isanr=False):

    img_input = Input(shape=(input_shape, input_shape, 3))

    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if _isseblock:
        x = _SE_block(x)
    elif _iscbamblock:
        x = _CBAM_block(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if _isseblock:
        x = _SE_block(x)
    elif _iscbamblock:
        x = _CBAM_block(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    if _isseblock:
        x = _SE_block(x)
    elif _iscbamblock:
        x = _CBAM_block(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', use_bias=False)(x)
    x = Dense(1024, activation='relu', use_bias=False)(x)
    x = Dense(classes, activation='softmax', use_bias=False)(x)

    model = Model(img_input, x)
    return model