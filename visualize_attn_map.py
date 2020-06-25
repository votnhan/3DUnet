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

from unet3d.model import attention_isensee2017_model
from segmentation import get_subject_tensor, resize_modal_image, normalize_data

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

def prepare_input(subject_fd, config):
    subject_name = os.path.basename(subject_fd)
    image_mris, original_affine, foreground = get_subject_tensor(subject_fd, 
                                                               subject_name)

    target_shape = tuple(config['inference_shape'])
    subject_data_fixed_size, affine = resize_modal_image(image_mris, target_shape)
    subject_tensor = normalize_data(subject_data_fixed_size)
    subject_tensor = np.expand_dims(subject_tensor, axis=0)
    return subject_tensor, original_affine, foreground


def prepare_model(config):
    model_path = config['model_file']
    model_graph = attention_isensee2017_model(input_shape=config["input_shape"], 
                                                n_labels=config["n_labels"], 
                                                n_base_filters=config["n_base_filters"], 
                                                activation_name='softmax', visualize=True)
    model_graph.load_weights(model_path)

    return model_graph


def save_npz_tensors(list_tensor, list_filenames, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i, tensor in enumerate(list_tensor):
        output_file = os.path.join(output_path, list_filenames[i])
        np.savez_compressed(output_file, tensor)

    
def get_intermidiate_feature(config, subject_fd, output_path):
    subject_tensor, original_affine, foreground = prepare_input(subject_fd, config)
    model = prepare_model(config)
    outputs = model.predict(subject_tensor)
    # Hard code temporary
    list_filenames = config['visualization']
    for k, v in list_filenames.items():
        filenames = v['filenames']
        range_idx = slice(v['range'][0], v['range'][1])
        list_tensors = outputs[range_idx]
        save_npz_tensors(list_tensors, filenames, output_path)
    
    print('Done getting intermidiate feature !')

parser = argparse.ArgumentParser(description='Get intermidiate feature map')
parser.add_argument('--subject_fd', type=str, 
                    default='brats/data/Train/HGG/Brats18_TCIA01_131_1')

parser.add_argument('--output_path', type=str, default='output')
parser.add_argument('--config_file', type=str, default='brats/config.json')


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.config_file, 'r') as cfg:
        config = json.load(cfg)
    
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["inference_shape"]))
    get_intermidiate_feature(config, args.subject_fd, args.output_path)