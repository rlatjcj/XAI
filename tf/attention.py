import keras
import keras.backend as K
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Reshape
from keras.layers import Add
from keras.layers import Activation
from keras.layers import Multiply
from keras.layers import Lambda
from keras.layers import Concatenate
from keras.layers import Conv2D

__all__ = ["_SE_block", "_CBAM_block"]

def _SE_block(_inputs, _ratio=16):
    """J. Hu et al., Squeeze-and-Excitation Networks, CVPR 2018.
    Arguments:
        _inputs : the feature map which want to be attended by the seblock.
        _ratio : the ratio for squeezing (defalut: 16).
    
    Returns:
        _x : the feature map attended by the seblock.
    """
    _channel = K.int_shape(_inputs)[-1]
    _x = GlobalAveragePooling2D()(_inputs)
    _x = Dense(int(_channel//_ratio), use_bias=False)(_x)
    _x = Activation('relu')(_x)
    _x = Dense(_channel, use_bias=False)(_x)
    _x = Activation('sigmoid')(_x)
    _x = Reshape((1, 1, _channel))(_x)
    _x = Multiply()([_x, _inputs])
    return _x


def _CBAM_block(_inputs, _ratio=8):
    """S. Woo et al., CBAM: Convolutional Block Attention Module, ECCV 2018.
    
    Arguments:
        _inputs : the feature map which want to be attended by the CBAM block.
        _ratio : the ratio for squeezing the fc layer at the part of the channel attention (default: 8).
        
    Returns:
        _x : the feature map which is attended by the CBAM block.
    """
    def _channel_attention(_inputs, _ratio=8):
        _channel = K.int_shape(_inputs)[-1]
        _share_layer_1 = Dense(_channel//_ratio, activation='relu')
        _share_layer_2 = Dense(_channel)

        _avg_pool = GlobalAveragePooling2D()(_inputs)
        _avg_pool = Reshape((1, 1, _channel))(_avg_pool)
        _avg_pool = _share_layer_1(_avg_pool)
        _avg_pool = _share_layer_2(_avg_pool)

        _max_pool = GlobalMaxPooling2D()(_inputs)
        _max_pool = Reshape((1, 1, _channel))(_max_pool)
        _max_pool = _share_layer_1(_max_pool)
        _max_pool = _share_layer_2(_max_pool)

        _combined_feature = Add()([_avg_pool, _max_pool])
        _combined_feature = Activation('sigmoid')(_combined_feature)
        _output = Multiply()([_combined_feature, _inputs])

        return _output

    def _spatial_attention(_inputs):
        _channel = K.int_shape(_inputs)[-1]

        _avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(_inputs)
        _max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(_inputs)
        _combined_feature = Concatenate(axis=-1)([_avg_pool, _max_pool])
        _combined_feature = Conv2D(1, (7, 7), padding='same', activation='sigmoid', use_bias=False)(_combined_feature)
        _output = Multiply()([_combined_feature, _inputs])

        return _output

    _x = _channel_attention(_inputs, _ratio)
    _x = _spatial_attention(_x)
    return _x


## P. Rodriguez et al.,
## Attend and Rectify: a Gated Attention Mechanism for Fine-Grained Recovery,
## ECCV 2018.

def _attention_head(_inputs, _out_channel):
    _x = Conv2D(_out_channel, (3, 3), padding='same', kernel_initializer='he_normal')(_inputs)
    _x = Activation('softmax', axis=-1)(_x)
    return _x
    
def _output_head(_inputs, _out_channel, _attention_head):
    _x = Conv2D(_out_channel, (3, 3), padding='same', kernel_initializer='he_normal')(_inputs)
    _x = Multiply()([_attention_head, _x])
    _x = GlobalAveragePooling2D()(_x)
    return _x
    
def _attention_gate(_inputs, _out_channel, _attention_head):
    _x = _output_head(_inputs, _out_channel, _attention_head)
    _x = Activation('tanh')(_x)
    _x = Activation('softmax')(_x)
    return _x
  
    
