import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras import backend as K
import numpy as np

class Normalize(Layer):
    
    def __init__(self, gamma_init=20, **kwargs):
        if K.image_data_format() == "channels_last":
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(Normalize, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis], ))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self._trainable_weights = [self.gamma]
        super(Normalize, self).build(input_shape)
    
    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        return output * self.gamma
    
    def get_config(self):
        config = {
            'gamma_init': self.gamma_init
        }
        base_config = super(Normalize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

