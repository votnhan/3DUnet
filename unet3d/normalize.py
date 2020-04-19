import os

import numpy as np
from nilearn.image import new_img_like

from unet3d.utils.utils import resize, read_image_files
from .utils import crop_img, crop_img_to, read_image


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    # As return_slices=True, crop_img here is just to get slices parameters, do not get cropped image
    return crop_img(img=foreground, return_slices=True, copy=True)

# crop = True!
def reslice_image_set(in_files, image_shape, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(image_files=in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    # loop over modalities
    for i, image_file in enumerate(set_of_files):
        # read_image read MRI image from path image_file
        image = read_image(in_file=image_file)
        # find the voxel has a value other than 0
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        # Convert true-false to 1-0
        return new_img_like(ref_niimg=image, data=foreground)
    else:
        return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        # Find means, stds of 4 modality for each sample in training set
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    # Find average means, stds over all training set => mean, std have shape: (1, 4)
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    # Normalize modalities of subject over all training set use calculated means, stds
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


def normalize_data_storage_independent(data_storage):
    
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        brain_masks = [data[i] > 0 for i in range(data.shape[0])]
        brain_data = [data[i][x] for i, x in enumerate(brain_masks)]
        mean = np.asarray([x.mean() for x in brain_data])
        std = np.asarray([x.std() for x in brain_data])
        for i, mask in enumerate(brain_masks):
            data_storage[index][i][mask] -= mean[i]
            data_storage[index][i][mask] /= std[i]
    
    return data_storage




