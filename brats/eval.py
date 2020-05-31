import sys
sys.path.append('../')

from unet3d.training import load_old_model
from unet3d.utils import pickle_load
from unet3d.data import open_data_file
from unet3d.generator import data_generator
import argparse
import json
import math

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
# print("tf.__version__ is", tf.__version__)
# print("tf.keras.__version__ is:", tf.keras.__version__)

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

def main(config=None):
    model = load_old_model(config, re_compile=False)
    data_file_opened = open_data_file(config["data_file"])
    validation_idxs = pickle_load(config['validation_file'])
    validation_generator = data_generator(data_file_opened, validation_idxs, 
                                        batch_size=config['validation_batch_size'], 
                                        n_labels=config['n_labels'], labels=config['labels'],
                                        skip_blank=config['skip_blank'], shuffle_index_list=False)
    steps = math.ceil(len(validation_idxs) / config['validation_batch_size'])
    results = model.evaluate(validation_generator, steps=steps, verbose=1)
    metrics_names = model.metrics_names
    for i, x in enumerate(metrics_names):
        print('{}: {}'.format(x, results[i]))
        
    data_file_opened.close()

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Residual 3D U-net For Brain Tumor Segmentation')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: config.json)')
    args = args.parse_args()
    
    config_file = args.config
    with open(config_file, 'r') as cfg:
        config = json.load(cfg)

    for key in config["keys_tuple"]:
        config[key] = tuple(config[key])

    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
    
    main(config)