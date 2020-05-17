import os
import glob
import json
import argparse

import sys
sys.path.append('../')

from unet3d.data import write_data_to_file, open_data_file
from unet3d.generator import get_training_and_validation_generators
from unet3d.model import isensee2017_model
from unet3d.training import load_old_model, train_model
import unet3d.metrics as module_metric
import keras.optimizers as opts

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

def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    subject_paths = glob.glob(os.path.join(os.path.dirname(__file__), "data", "Train", "*", "*"))
    subject_paths.sort()
    for subject_dir in subject_paths:
        subject_dir_name = os.path.basename(subject_dir)
        subject_ids.append(subject_dir_name)
        subject_files = list()
        
        for modality in config["all_modalities"] + config["label_modality"]:
            subject_files.append(os.path.join(subject_dir, subject_dir_name + '_' + modality + ".nii.gz"))
        
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files

def main(config=None):
    # convert input images into an hdf5 file
    overwrite = config['overwrite']
    if overwrite or not os.path.exists(config["data_file"]):
        training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

        write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
                           subject_ids=subject_ids, norm_type=config['normalization_type'])
    data_file_opened = open_data_file(config["data_file"])

    if not overwrite and os.path.exists(config["model_file"]):
        model = load_old_model(config, re_compile=False)
    else:
        # instantiate new model
        model = isensee2017_model(input_shape=config["input_shape"], 
                                    n_labels=config["n_labels"], 
                                    n_base_filters=config["n_base_filters"], 
                                    activation_name='softmax')

        optimizer = getattr(opts, config["optimizer"]["name"])(**config["optimizer"].get('args'))
        loss = getattr(module_metric, config["loss_fc"])
        metrics = [getattr(module_metric, x) for x in config["metrics"]]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # get training and testing generators
    train_generator, validation_generator, n_train_steps, n_validation_steps = get_training_and_validation_generators(
        data_file_opened,
        batch_size=config["batch_size"],
        data_split=config["validation_split"],
        overwrite=overwrite,
        validation_keys_file=config["validation_file"],
        training_keys_file=config["training_file"],
        n_labels=config["n_labels"],
        labels=config["labels"],
        validation_batch_size=config["validation_batch_size"],
        permute=config["permute"],
        augment=config["augment"],
        skip_blank=config["skip_blank"],
        augment_flip=config["flip"],
        augment_distortion_factor=config["distort"])

    # run training
    train_model(model=model,
                model_file=config["model_file"],
                training_generator=train_generator,
                validation_generator=validation_generator,
                steps_per_epoch=n_train_steps,
                validation_steps=n_validation_steps,
                initial_learning_rate=config["optimizer"]["args"]["lr"],
                learning_rate_drop=config["learning_rate_drop"],
                learning_rate_patience=config["patience"],
                early_stopping_patience=config["early_stop"],
                n_epochs=config["n_epochs"],
                model_best_path=config['model_best'])
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
