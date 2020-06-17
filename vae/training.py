import csv
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from unet3d.training import get_callbacks, train_model
from keras.utils.io_utils import h5dict
import vae.metrics as module_metric
import keras.optimizers as opts
import json
import warnings


K.common.set_image_dim_ordering('th')

def load_old_model(config, re_compile=False):
    print("Loading pre-trained model")
    model_cfg = config['model']
    trainer_cfg = config['trainer']

    custom_objects = dict()
    from vae.model import GroupNormalization
    custom_objects['GroupNormalization'] = GroupNormalization
    model = load_model(model_cfg['model_file'], custom_objects=custom_objects, compile=False)
    idx_mean, idx_var = model_cfg['idxs_mean_var']
    z_mean, z_var = model.layers[idx_mean], model.layers[idx_var]
    # Prepare loss
    loss = module_metric.loss_VAE(model_cfg['input_shape'], z_mean, z_var, 
                                    trainer_cfg['weight_L2'], trainer_cfg['weight_KL'])
    # Prepare metrics
    metrics = [getattr(module_metric, metric) for metric in trainer_cfg['metrics']]
    # Prepare optimizer
    checkpoint = h5dict(model_cfg['model_file'], 'r')
    training_config = checkpoint.get('training_config')
    training_config = json.loads(training_config.decode('utf-8'))
    optimizer_config = training_config['optimizer_config']
    optimizer = opts.deserialize(optimizer_config, custom_objects=dict())
    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if 'optimizer_weights' in checkpoint:
        # Build train function (to get weight updates).
        model._make_train_function()
        optimizer_weights_group = checkpoint['optimizer_weights']
        optimizer_weight_names = [
            n.decode('utf8') for n in
            optimizer_weights_group['weight_names']]
        optimizer_weight_values = [optimizer_weights_group[n] for n in
                                    optimizer_weight_names]
        try:
            model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
            warnings.warn('Error in loading the saved optimizer '
                            'state. As a result, your model is '
                            'starting with a freshly initialized '
                            'optimizer.')

    return model

        