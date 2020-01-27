import pickle
import os
import collections

import nibabel as nib
import numpy as np
from nilearn.image import reorder_img, new_img_like

from .nilearn_custom_utils.nilearn_utils import crop_img_to
from .sitk_utils import resample_to_spacing, calculate_origin_offset


def pickle_dump(item, out_file):
    with open(out_file, "wb") as opened_file:
        pickle.dump(item, opened_file)


def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)


def get_affine(in_file):
    return read_image(in_file).affine


def read_image_files(image_files, image_shape=None, crop=None, label_indices=None):
    """
    :param image_files: list of files MRI image
    :param image_shape: shape of MRI image
    :param crop: range of x, y, z for crop
    :param label_indices: index of label path in image_files
    :return: list of MRI images
    """
    if label_indices is None:
        label_indices = []
    elif not isinstance(label_indices, collections.Iterable) or isinstance(label_indices, str):
        label_indices = [label_indices]
    image_list = list()
    for index, image_file in enumerate(image_files):
        if index in label_indices:
            interpolation = "nearest"
        else:
            interpolation = "linear"
        image_list.append(read_image(image_file, image_shape=image_shape, crop=crop, interpolation=interpolation))

    return image_list

# Read MRI image then crop and resize.
def read_image(in_file, image_shape=None, interpolation='linear', crop=None):
    print("Reading: {}".format(in_file))
    image = nib.load(os.path.abspath(in_file))
    image = fix_shape(image)
    if crop:
        image = crop_img_to(image, crop, copy=True)
    if image_shape:
        return resize(image, new_shape=image_shape, interpolation=interpolation)
    else:
        return image


def fix_shape(image):
    if image.shape[-1] == 1:
        return image.__class__(dataobj=np.squeeze(image.get_data()), affine=image.affine)
    return image

# Resize image to new_shape use library SimpleITK
def resize(image, new_shape, interpolation="linear"):
    image = reorder_img(image, resample=interpolation)
    zoom_level = np.divide(new_shape, image.shape)
    new_spacing = np.divide(image.header.get_zooms(), zoom_level)
    new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
                                   interpolation=interpolation)
    new_affine = np.copy(image.affine)
    np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
    new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())
    return new_img_like(image, new_data, affine=new_affine)
