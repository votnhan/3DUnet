from functools import partial

from keras import backend as K
from keras import losses
import tensorflow.math as tfmth


def dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=1.):
    y_true_f = K.cast(y_true, dtype='float32')
    y_pred_f = K.cast(y_pred, dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f, axis=axis)
    sum_true = K.sum(y_true_f, axis=axis)
    sum_pred = K.sum(y_pred_f, axis=axis)
    # this formula need to be proved !
    dice = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return K.mean(dice) 


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    dice_score_each_class = 2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth)

    return K.mean(dice_score_each_class)


def weighted_dice_coefficient_loss(y_true, y_pred):
    return 1 - weighted_dice_coefficient(y_true, y_pred)


def dice_and_entropy_combination_loss(y_true, y_pred):
    return .99*weighted_dice_coefficient_loss(y_true, y_pred) + .01*losses.categorical_crossentropy(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def get_one_hot_from_output(y_true, y_pred):
    cls_true = K.argmax(y_true, axis=-4)
    cls_pred = K.argmax(y_pred, axis=-4)
    return cls_true, cls_pred


def dice_whole_tumor(y_true, y_pred):
    mask_true, mask_pred = get_one_hot_from_output(y_true, y_pred)
    mask_true = tfmth.greater(mask_true, 0)
    mask_pred = tfmth.greater(mask_pred, 0)
    return dice_coefficient(mask_true, mask_pred)


def dice_tumor_core(y_true, y_pred):
    mask_true, mask_pred = get_one_hot_from_output(y_true, y_pred)
    mask_true = tfmth.logical_or(tfmth.equal(mask_true, 1), tfmth.equal(mask_true, 3))
    mask_pred = tfmth.logical_or(tfmth.equal(mask_pred, 1), tfmth.equal(mask_pred, 3))
    return dice_coefficient(mask_true, mask_pred)


def dice_enhancing_tumor(y_true, y_pred):
    mask_true, mask_pred = get_one_hot_from_output(y_true, y_pred)
    mask_true = tfmth.equal(mask_true, 3)
    mask_pred = tfmth.equal(mask_pred, 3)
    return dice_coefficient(mask_true, mask_pred)

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
