from numpy.lib.function_base import append, flip
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.python.keras.backend import var
from tensorflow.python.keras.engine.input_layer import Input


class AnchorBoxes(Layer):
    
    def __init__(self,  img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, clip=True,variances=[0.1],**kwargs):

        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.flip = flip
        self.variances = np.array(variances)
        self.clip = clip
        super(AnchorBoxes, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)
    
    def call(self, x, mask=None):
        
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            if isinstance(x, list):
                input_shape = K.int_shape(x[0])
            else:
                input_shape = K.int_shape(x)

        feature_map_width = input_shape[2]
        feature_map_height = input_shape[1]
        img_width = self.img_size[1]
        img_height = self.img_size[0]
        box_heights = []
        box_widths = []    

        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 0:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        
        box_widths = np.array(box_widths) * 0.5
        box_heights = np.array(box_heights) * 0.5
        
        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        step_height = img_height / feature_map_height
        step_width = img_width / feature_map_width
        

        linx = np.linspace(0.5 * step_width, img_width - 0.5 *step_width,
                           feature_map_height)
        liny = np.linspace(0.5 * step_height, img_height - 0.5 * step_height,
                           feature_map_height)
        center_x, center_y = np.meshgrid(linx, liny)
        center_x = center_x.reshape(-1, 1)
        center_y = center_y.reshape(-1, 1)

        # Every prior_boxes need two boxes, one is be used (xmin, ymin), the other is be used (xmax, ymax) 
        num_priors = len(self.aspect_ratios)
        prior_boxes = np.concatenate((center_x, center_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))
        
        # Compute four corners
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        # Normalize
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)

        num_boxes = len(prior_boxes)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = K.expand_dims(tf.cast(prior_boxes, dtype=tf.float32), 0)
    
        pattern = [tf.shape(x)[0], 1, 1]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)

        return prior_boxes_tensor

    def compute_output_shape(self, input_shape):
        if K.image_data_format() == "channels_last":
            batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        else: # Not yet relevant since TensorFlow is the only supported backend right now, but it can't harm to have this in here for the future
            batch_size, feature_map_channels, feature_map_height, feature_map_width = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)
    
    def get_config(self):
       
        config = {
            'img_size': self.img_size,
            'min_size': self.min_size,
            'max_size': self.max_size,
            'aspect_ratios': list(self.aspect_ratios),
            'flip': self.flip,
            'clip': self.clip,
            'variances': list(self.variances)
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
