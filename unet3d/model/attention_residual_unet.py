from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, \
                            SpatialDropout3D, Conv3D, Concatenate, \
                            Softmax, Multiply
from keras.engine import Model
from .unet import create_convolution_block, concatenate
from keras.regularizers import l2


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def attention_isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, activation_name="sigmoid", visualize=False):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    if visualize:
        before_masked = list()
        gatting_signal = list()
        masks = list()
        after_masked = list()
        normed_after_masked = list()

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])

        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    gatting = create_gatting_signal(current_layer, level_filters[-2])
    
    if visualize:
        before_masked = level_output_layers[:-1].copy()
        before_masked.reverse()
        gatting_signal.append(gatting)

    for level_number in range(depth - 2, -1, -1):
        if level_number > 0:
            up_size = 3 * (2 ** (depth - 1 - level_number), )
            masked_current_layer, mask, masked_x = create_attention_gate(level_output_layers[level_number], gatting, 
                                                            level_filters[level_number], up_size)
            if visualize:
                normed_after_masked.append(masked_current_layer)
                masks.append(mask)
                after_masked.append(masked_x)

        else:
            masked_current_layer = level_output_layers[level_number]
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([masked_current_layer, up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = None

    if activation_name == 'sigmoid':
        activation_block = Activation(activation_name)(output_layer)
    elif activation_name == 'softmax':
        activation_block = Softmax(axis=1)(output_layer)

    
    if visualize:
        outputs = before_masked + gatting_signal + masks + after_masked + normed_after_masked
        outputs.append(activation_block)
    else:
        outputs = activation_block
        
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_conv3D(output_channels, kernel_size, strides=(1, 1, 1), use_bias=False):
    conv3d = Conv3D(output_channels, kernel_size, use_bias=use_bias)
    return conv3d


def create_instance_norm(axis=1):
    from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
    ins_norm = InstanceNormalization(axis=1)
    return ins_norm


def create_attention_gate(x, gating, inter_channels, up_size=(2, 2, 2), masks_list=None):
    w_x = create_conv3D(inter_channels, (1, 1, 1), strides=(1, 1, 1)) (x)
    w_gating = create_conv3D(inter_channels, (1, 1, 1), use_bias=True)(gating)
    up_gatting = UpSampling3D(size=up_size)(w_gating)
    sum_x_gating = Add()([w_x, up_gatting])
    relu_sum = Activation('relu')(sum_x_gating)
    psi = create_conv3D(1, (1, 1, 1), use_bias=True)(relu_sum)
    mask = Activation('sigmoid')(psi)
    masked_x = Multiply()([x, mask])
    W = create_instance_norm(axis=1)(create_conv3D(x.shape[1], (1, 1, 1))(masked_x))
    return Activation('relu')(W), mask, masked_x


def create_gatting_signal(center, out_size, is_norm=True):
    x = create_conv3D(out_size, (1, 1, 1))(center)
    
    if is_norm:
        x = create_instance_norm(axis=1)(x)
    
    relu_x = Activation('relu')(x)
    return relu_x


def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2



