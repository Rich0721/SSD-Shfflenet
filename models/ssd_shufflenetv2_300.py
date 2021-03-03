import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Activation, Conv2D, Reshape, Concatenate, BatchNormalization, Add, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.layers import ReLU, Flatten, MaxPooling2D, ZeroPadding2D
from ssd_keras_layers.anchorBoxes import AnchorBoxes
from ssd_keras_layers.normalize import Normalize

from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def channel_split(inputs, name=''):

    channels = inputs.shape.as_list()[-1]
    split_channel = channels // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:split_channel], name="{}/split_{}slice".format(name, 0))(inputs)
    c = Lambda(lambda z: z[:, :, :, split_channel:], name="{}/split_{}slice".format(name, 1))(inputs)

    return c_hat, c


def channel_shuffle(inputs):
    height, width, channels = inputs.shape.as_list()[1:]
    channel_per_split = channels // 2
    x = K.reshape(inputs, [-1, height, width, 2, channel_per_split])
    x = K.permute_dimensions(x, (0, 1, 2, 3, 4))
    x = K.reshape(x, [-1, height, width, channels])
    return x


def shuffle_unit(inputs, out_channels, bottleneck_ratio, strides=2, stage=1, block=1):

    if K.image_data_format() == 'channel_last':
        bn_axis = -1
    else:
        bn_axis = 1
    
    prefix = 'stage{}/block{}'.format(stage, block)

    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/split'.format(prefix))
        inputs = c
    
    bottlneck_channels = int(out_channels * bottleneck_ratio)

    x = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name="{}/conv1".format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv1_BN".format(prefix))(x)
    x = Activation("relu", name="{}/conv1_relu".format(prefix))(x)

    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', name='{}/dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name='{}/dwconv_BN'.format(prefix))(x)
    x = Activation("relu", name='{}/dwconv_relu'.format(prefix))(x)

    x = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name='{}/conv2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv2_BN".format(prefix))(x)
    x = Activation("relu", name="{}/conv2_relu".format(prefix))(x)

    if strides < 2:
        ret = Concatenate(name="{}/concat".format(prefix))([x, c_hat])
    else:
        shortcut = DepthwiseConv2D((3, 3), strides=strides, padding='same', name='{}/dwconv2'.format(prefix))(inputs)
        shortcut = BatchNormalization(axis=bn_axis, name='{}/dwconv2_BN'.format(prefix))(shortcut)

        shortcut = Conv2D(bottlneck_channels, (1, 1), strides=1, padding='same', name='{}/conv3'.format(prefix))(shortcut)
        shortcut = BatchNormalization(axis=bn_axis, momentum=0.999, epsilon=1e-3, name="{}/conv3_BN".format(prefix))(shortcut)
        shortcut = Activation("relu", name='{}/conv3_relu'.format(prefix))(shortcut)
        ret = Concatenate(name="{}/concat".format(prefix))([x, shortcut])
    
    
    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)
    return ret


def block(inputs, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(inputs, out_channels=channel_map[stage-1], bottleneck_ratio=bottleneck_ratio, strides=2, stage=stage, block=1)

    for i in range(1, repeat):
        x = shuffle_unit(x, out_channels=channel_map[stage-1], bottleneck_ratio=bottleneck_ratio, strides=1, stage=stage, block=i+1)
    return x

def ssd_conv(inputs, filters, kerner_size, padding='same', strides=(1, 1), name=None):

    x = Conv2D(filters, kerner_size, strides=strides, padding=padding, use_bias=False, name=name)(inputs)
    x = BatchNormalization(momentum=0.999, epsilon=1e-3, name=name+"/BN")(x)
    x = ReLU(6.)(x)
    return x

def SSD300(img_size, n_classes, l2_regularization=5e-4,
            anchors=[30, 60, 111, 162, 213, 264, 315],
            variances=[0.1, 0.1, 0.2, 0.2]):
    
    classes = n_classes + 1# Account for the background class.
    n_boxes = [4, 6, 6, 6, 4, 4]


    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = Input(shape=(img_size))
    
    num_shuffle_units = [3, 7, 3]
    scale_factor=1.0
    bottleneck_ratio=1
    dims = {0.5:48, 1:116, 1.5:176, 2:244}
    
    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype=np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= dims[bottleneck_ratio]
    out_channels_in_stage[0] = 24
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    
    stage1 = Conv2D(out_channels_in_stage[0], (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='relu', name='conv1')(x)
    stage1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1')(stage1)

    stage2 = block(stage1, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, stage=2)
    stage3 = block(stage2, out_channels_in_stage, repeat=7, bottleneck_ratio=bottleneck_ratio, stage=3)
    stage4 = block(stage3, out_channels_in_stage, repeat=3, bottleneck_ratio=bottleneck_ratio, stage=4)

    print("Stage:{}".format(stage4.shape))
    '''
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(stage4)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv6_2')(conv6_1)
    
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_1)
    
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv8_2')(conv8_1)
    
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same',  name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(256, (3, 3), activation='relu', padding='valid', name='conv9_2')(conv9_1)
    '''
    conv6_1 = ssd_conv(stage4, 256, (1, 1), padding='same', name='conv6_1')
    conv6_2 = ssd_conv(conv6_1, 512, (3, 3), strides=(2, 2), padding='same', name='conv6_2')
    
    conv7_1 = ssd_conv(conv6_2, 128, (1, 1), padding='same', name='conv7_1')
    conv7_2 = ssd_conv(conv7_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv7_2')
    
    conv8_1 = ssd_conv(conv7_2, 128, (1, 1), padding='same', name='conv8_1')
    conv8_2 = ssd_conv(conv8_1, 256, (3, 3), strides=(2, 2), padding='same', name='conv8_2')
    
    conv9_1 = ssd_conv(conv8_2, 64, (1, 1), padding='same', name='conv9_1')
    conv9_2 = ssd_conv(conv9_1, 128, (3, 3), strides=(2, 2), padding='same', name='conv9_2')
    
    # Build SSD network


    conv4_3_norm = Normalize(20, name='conv4_3_norm')(stage3)

    # Build "n_classes" confidence values for each box. Ouput shape: (b, h, w, n_boxes*n_classes)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * classes, (3, 3), padding='same',  name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    fc7_mbox_conf = Conv2D(n_boxes[1] * classes, (3, 3), padding='same', name='fc7_mbox_conf')(stage4)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(conv6_2)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * classes, (3, 3), padding='same',  name='conv7_2_mbox_conf')(conv7_2)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * classes, (3, 3), padding='same', name='conv8_2_mbox_conf')(conv8_2)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * classes, (3, 3), padding='same', name='conv9_2_mbox_conf')(conv9_2)

    # Build 4 box coordinates for each box. Output shape: (b, h, w, n_boxes * 4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', name='fc7_mbox_loc')(stage4)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same',  name='conv7_2_mbox_loc')(conv7_2)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', name='conv9_2_mbox_loc')(conv9_2)
    
    # Generate the anchor boxes. Output shape: (b, h, w, n_boxes, 8)
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[0], max_size=anchors[1],aspect_ratios=[2],
                                             variances=variances,name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    fc7_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[1], max_size=anchors[2],aspect_ratios=[2,3],
                                             variances=variances,name='fc7_mbox_priorbox')(stage4)
    conv6_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[2], max_size=anchors[3],aspect_ratios=[2, 3],
                                             variances=variances, name='conv6_2_mbox_priorbox')(conv6_2)
    conv7_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[3], max_size=anchors[4],aspect_ratios=[2, 3],
                                             variances=variances, name='conv7_2_mbox_priorbox')(conv7_2)
    conv8_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[4], max_size=anchors[5],aspect_ratios=[2],
                                             variances=variances, name='conv8_2_mbox_priorbox')(conv8_2)
    conv9_2_mbox_priorbox = AnchorBoxes(img_size=img_size, min_size=anchors[5], max_size=anchors[6],aspect_ratios=[2],
                                        variances=variances, name='conv9_2_mbox_priorbox')(conv9_2)

    ### Reshape

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Flatten(name='conv4_3_norm_mbox_conf_reshape')(conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Flatten(name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Flatten(name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Flatten(name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Flatten(name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Flatten(name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    conv4_3_norm_mbox_loc_reshape = Flatten(name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Flatten(name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Flatten(name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Flatten(name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Flatten(name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Flatten(name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)

    ### Concatenate the predictions from the different layers

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox,
                                                               fc7_mbox_priorbox,
                                                               conv6_2_mbox_priorbox,
                                                               conv7_2_mbox_priorbox,
                                                               conv8_2_mbox_priorbox,
                                                               conv9_2_mbox_priorbox])
    
    mbox_loc = Reshape((-1, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((-1, classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)

    predictions = Concatenate(axis=2, name='predictions')([mbox_loc,
                            mbox_conf,
                            mbox_priorbox])
    
    
    model = Model(inputs=x, outputs=predictions)
    return model



