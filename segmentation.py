import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import copy
import argparse
import glob
sys.path.append('./')

from unet3d.utils.utils import resize, read_image, crop_img_to, fix_shape
from unet3d.prediction import prediction_to_image
from unet3d.training import load_old_model

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus

original_shape = (240, 240, 155)
input_shape = (128, 128, 128)
extension = '.nii.gz'
all_modalities = ('t1', 't1ce', 'flair','t2')
labels = (0, 1, 2, 4)
background_value = 0
tolerance = 0.00001
rtol = 1e-8

def get_subject_tensor(subject_folder, subject_name):
    input_mris = []
    affine = None
    for i, modal in enumerate(all_modalities):
        path_modal = os.path.join(subject_folder, subject_name+'_'+modal + extension)
        modal_image = nib.load(path_modal)
        modal_tensor = modal_image.get_data()
        is_foreground = np.logical_or(modal_tensor < (background_value - tolerance),
                                      modal_tensor > (background_value + tolerance))
        
        if i == 0:
          foreground = np.zeros(is_foreground.shape, dtype=np.uint8)
          affine = modal_image.get_affine()

        input_mris.append(modal_image)
        foreground[is_foreground] = 1

    return input_mris, affine, foreground

def crop_subject_modals(subject_modal_imgs, input_shape, slices):
  subject_data = []
  affine = None
  for i, modal_img in enumerate(subject_modal_imgs):
    modal_img = fix_shape(modal_img)
    modal_img = crop_img_to(modal_img, slices, copy=True)
    new_img = resize(modal_img, new_shape=input_shape, interpolation='linear')
    subject_data.append(new_img.get_data())
    if i == 0:
      affine = new_img.get_affine()

  subject_data = np.asarray(subject_data)
  return subject_data, affine

def resize_modal_image(subject_modal_imgs, target_shape, interpolation='linear'):
  subject_data = []
  affine = None
  for i, modal_img in enumerate(subject_modal_imgs):
    modal_img = fix_shape(modal_img)
    new_img = resize(modal_img, new_shape=target_shape, interpolation=interpolation)
    subject_data.append(new_img.get_data())
    if i == 0:
      affine = new_img.get_affine()

  subject_data = np.asarray(subject_data)
  return subject_data, affine

def normalize_data(input_tensor):
    input_tensor = input_tensor.astype(np.float32)
    brain_mask = [input_tensor[i] > 0 for i in range(input_tensor.shape[0])]
    brain_data = [input_tensor[i][mask] for i, mask in enumerate(brain_mask)]
    
    means = np.asarray([x.mean() for x in brain_data])
    stds = np.asarray([x.std() for x in brain_data])

    result = copy.deepcopy(input_tensor)
    for i, mask in enumerate(brain_mask):
      result[i][mask] = (input_tensor[i][mask] - means[i]) / stds[i]

    return result

def predict(model, image_tensor, affine):
  result = model.predict(image_tensor)
  image_label = prediction_to_image(result, affine, label_map=True, labels=labels)
  return image_label

def restore_dimension(image_label, slices, affine):
  old_shape = tuple([x.stop - x.start for x in slices])
  old_cropped_image = resize(image_label, old_shape, interpolation='nearest')
  rotated_image = np.rot90(old_cropped_image.get_data(), 2)
  result = np.zeros(original_shape, dtype=np.uint8)
  tp_slices = tuple(slices)
  result[tp_slices] = rotated_image
  return nib.Nifti1Image(result, affine)

def get_slices(foreground):
  infinity_norm = max(-np.min(foreground), np.max(foreground))
  passes_threshold = np.logical_or(foreground < -rtol * infinity_norm,
                                     foreground > rtol * infinity_norm)
  coords = np.array(np.where(passes_threshold))
  start = coords.min(axis=1)
  end = coords.max(axis=1) + 1
  start = np.maximum(start - 1, 0)
  end = np.minimum(end + 1, foreground.shape[:3])
  slices = [slice(s, e) for s, e in zip(start, end)]
  return slices

def segmentation_for_set_patients(list_ids_file, path_dataset, config, output_path):
  file = open(list_ids_file, 'r')
  contents = file.read()
  list_ids = contents.split('\n')
  file.close()
  pattern = os.path.join(path_dataset, '*', '*')
  list_paths = glob.glob(pattern)
  for idx, ids in enumerate(list_ids):
    for path_subject in list_paths:
      if ids in path_subject:
        segmentation_for_patient(path_subject, config, output_path)
        break
    
    print('Done {}/{} patients'.format(idx+1, len(list_ids)))

  print('Done for dataset: {}'.format(path_dataset))


def segmentation_for_patient(subject_fd, config, output_path, mode='size_same_input'):
  model = load_old_model(config)
  subject_name = os.path.basename(subject_fd)
  image_mris, original_affine, foreground = get_subject_tensor(subject_fd, 
                                                               subject_name)
  if mode == 'size_same_input':
    slices = get_slices(foreground)
    subject_data_fixed_size, affine = crop_subject_modals(image_mris, input_shape, 
                                                          slices)
  elif mode == 'size_interpolate':
    target_shape = tuple(config['inference_shape'])
    subject_data_fixed_size, affine = resize_modal_image(image_mris, target_shape)
  else:
    print('Do not support mode {} for inference'.format(mode))
    return
  
  subject_tensor = normalize_data(subject_data_fixed_size)

  subject_tensor = np.expand_dims(subject_tensor, axis=0)
  output_predict = predict(model, subject_tensor, affine)
  
  if mode == 'size_same_input':
    output = restore_dimension(output_predict, slices, original_affine)
  elif mode == 'size_interpolate':
    output = resize(output_predict, new_shape=original_shape, interpolation='nearest')
  else:
    print('Do not support mode {} for inference'.format(mode))
    return

  output_fd = os.path.join(output_path, subject_name)
  if not os.path.exists(output_fd):
    os.makedirs(output_fd)
  output_file = os.path.join(output_fd, '{}_prediction{}'.format(subject_name, 
                                                      extension))
  output.to_filename(output_file)
  
  print('Patient {} is done !'.format(subject_fd))

parser = argparse.ArgumentParser(description='Segment data')
parser.add_argument('--subject_fd', type=str, 
                    default='brats/data/Train/HGG/Brats18_TCIA01_131_1')

parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--config_file', type=str, default='brats/config.json')

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_file, 'r') as cfg:
        config = json.load(cfg)
    segmentation_for_patient(args.subject_fd, config, args.output_path)
    segmentation_for_set_patients('val_subject_name.txt', 'brats/data/Train', 
                              config, 'output/original_size')

