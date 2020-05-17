import os
import sys
import json

sys.path.append('../')

from unet3d.prediction import run_validation_cases

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

def prepare_config(config_file='config.json'):
    config_file = 'config.json'
    with open(config_file, 'r') as cfg:
        config = json.load(cfg)

    for key in config["keys_tuple"]:
        config[key] = tuple(config[key])
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
    return config

def main():

    config = prepare_config()
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(config=config, output_label_map=True, output_dir=prediction_dir)


if __name__ == "__main__":
    main()
