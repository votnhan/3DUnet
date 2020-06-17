import os
import json
import argparse
import sys
sys.path.append('../')
from vae.model import create_3d_VAE_model
from vae.generator import VAEGenerator
from vae.metrics import loss_VAE
import vae.metrics as module_metric
from vae.training import train_model, load_old_model
from unet3d.data import open_data_file
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

def main(config=None):
    # Prepare vae model
    model_cfg = config['model']
    model, z_mean, z_var = create_3d_VAE_model(model_cfg['input_shape'], model_cfg['num_filters_enc'], 
                                model_cfg['num_filters_dec'], model_cfg['size_vector_latent'])

    # Prepare trainer
    if os.path.exists(model_cfg['model_file']):
        model = load_old_model(config)
    else:
        trainer_cfg = config['trainer']
        optimizer = getattr(opts, config['optimizer']['name'])(**config['optimizer'].get('args'))
        loss = loss_VAE(model_cfg['input_shape'], z_mean, z_var, trainer_cfg['weight_L2'], trainer_cfg['weight_KL'])
        metrics = [getattr(module_metric, x) for x in trainer_cfg["metrics"]]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Prepare generator
    generator_cfg = config['generator']
    noise_h5_data = open_data_file(generator_cfg['noise_h5_file'])
    clean_h5_data = open_data_file(generator_cfg['clean_h5_file'])
    train_generator = VAEGenerator(generator_cfg['train_ids'], noise_h5_data, clean_h5_data, 
                        generator_cfg['train_batch_size'], generator_cfg['shuffe_train'])
    val_generator = VAEGenerator(generator_cfg['val_ids'], noise_h5_data, clean_h5_data, 
                        generator_cfg['val_batch_size'], False)

    train_model(model=model, model_file=model_cfg['model_file'], 
                training_generator=train_generator, validation_generator=val_generator, 
                steps_per_epoch=len(train_generator), validation_steps=len(val_generator),
                initial_learning_rate=config["optimizer"]["args"]["lr"], 
                learning_rate_drop=trainer_cfg['learning_rate_drop'], 
                learning_rate_patience=trainer_cfg["patience"],
                early_stopping_patience=trainer_cfg["early_stop"],
                n_epochs=trainer_cfg["n_epochs"],
                model_best_path=trainer_cfg['model_best'])

    noise_h5_data.close()
    clean_h5_data.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='3D VAE for denoising MRI image')
    args.add_argument('-c', '--config', default='cfg/vae/config.json', type=str,
                      help='config file path (default: config.json)')
    args = args.parse_args()
    
    config_file = args.config
    with open(config_file, 'r') as cfg:
        config = json.load(cfg)

    main(config)
