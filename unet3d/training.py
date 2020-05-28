import math
import keras.optimizers as opts
import runai.ga
import time
import six
import csv
import numpy as np
from collections import Iterable, OrderedDict
from functools import partial

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model

import unet3d.metrics as module_metric

from unet3d.model import unet_model_3d, isensee2017_model

K.common.set_image_dim_ordering('th')

class CSVLoggerTimeEpoch(CSVLogger):
    def __init__(self, filename, separator=',', append=False):
        super().__init__(filename, separator, append)
    
    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        print('Epoch {}, lr: {}.'.format(epoch, K.eval(self.model.optimizer.lr)))
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch', 'time_for_epoch'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        delta_time = time.time() - self.epoch_time_start
        row_dict.update({'time_for_epoch': delta_time})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()





# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))


def get_callbacks(model_file, initial_learning_rate=0.0001, learning_rate_drop=0.5, learning_rate_epochs=None,
                  learning_rate_patience=50, logging_file="training.log", verbosity=1,
                  early_stopping_patience=None, model_best_path='checkpoints/model_best.h5'):
    callbacks = list()
    callbacks.append(ModelCheckpoint(model_file, save_best_only=False, verbose=1))
    callbacks.append(CSVLoggerTimeEpoch(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs)))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))
    
    callbacks.append(ModelCheckpoint(model_best_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min'))
    return callbacks


def load_old_model(config, re_compile=False):
    print("Loading pre-trained model")
    custom_objects = dict()
    custom_objects[config['loss_fc']] = getattr(module_metric, config['loss_fc'])
    for metric in config['metrics']:
        custom_objects[metric] = getattr(module_metric, metric)
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        model = load_model(config['model_file'], custom_objects=custom_objects)
        if re_compile:
            optimizer = getattr(opts, config["optimizer"]["name"])(**config["optimizer"].get('args'))
            
            if config['accumulation_step'] > 1:
                optimizer = runai.ga.keras.optimizers.Optimizer(optimizer, 
                                    steps=config['accumulation_step'])
            
            loss = getattr(module_metric, config["loss_fc"])
            metrics = [getattr(module_metric, x) for x in config['metrics']]
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps,
                initial_learning_rate=0.001, learning_rate_drop=0.5, learning_rate_epochs=None, n_epochs=500,
                learning_rate_patience=20, early_stopping_patience=None, model_best_path=None):
    """
    Train a Keras model.
    :param early_stopping_patience: If set, training will end early if the validation loss does not improve after the
    specified number of epochs.
    :param learning_rate_patience: If learning_rate_epochs is not set, the learning rate will decrease if the validation
    loss does not improve after the specified number of epochs. (default is 20)
    :param model: Keras model that will be trained.
    :param model_file: Where to save the Keras model.
    :param training_generator: Generator that iterates through the training data.
    :param validation_generator: Generator that iterates through the validation data.
    :param steps_per_epoch: Number of batches that the training generator will provide during a given epoch.
    :param validation_steps: Number of batches that the validation generator will provide during a given epoch.
    :param initial_learning_rate: Learning rate at the beginning of training.
    :param learning_rate_drop: How much at which to the learning rate will decay.
    :param learning_rate_epochs: Number of epochs after which the learning rate will drop.
    :param n_epochs: Total number of epochs to train the model.
    :return: 
    """
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        workers=0,
                        callbacks=get_callbacks(model_file,
                                                initial_learning_rate=initial_learning_rate,
                                                learning_rate_drop=learning_rate_drop,
                                                learning_rate_epochs=learning_rate_epochs,
                                                learning_rate_patience=learning_rate_patience,
                                                early_stopping_patience=early_stopping_patience,
                                                model_best_path=model_best_path))
