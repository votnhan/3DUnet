import numpy as np
from keras.utils import Sequence
from utils import pickle_load
from math import ceil
from random import shuffle
from copy import deepcopy


class VAEGenerator(Sequence):
    def __init__(self, ids_file, noise_opened_file, clean_opened_file, 
                    batch_size=1, shuffle=True):
        self.ids_file = ids_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.noise_opened_file = noise_opened_file
        self.clean_opened_file = clean_opened_file
        self.ids = pickle_load(self.ids_file)
        self.ids_clone = deepcopy(self.ids)
        self.num_iters = int(ceil(len(self.ids) / self.batch_size))
        self.on_epoch_end()

    
    def __len__(self):
        return self.num_iters

    
    def on_epoch_end(self):
        if self.shuffle:
            shuffle(self.ids)


    def __getitem__(self, index):
        lower = index * self.batch_size
        upper = (index + 1) * self.batch_size
        if upper > len(self.ids_clone):
            upper = len(self.ids_clone)

        indexes = self.ids[lower: upper]
        X, Y = self.__data_generation(indexes)
        return X, Y


    def __data_generation(self, indexes):
        X = list()
        Y = list()
        for idx in indexes:
            x = np.expand_dims(self.noise_opened_file.root.data[idx], axis=0)
            y = np.expand_dims(self.clean_opened_file.root.data[idx], axis=0)
            X.append(x)
            Y.append(y)

        X_ = np.concatenate(X, axis=0)
        Y_ = np.concatenate(Y, axis=0)
        return X_, Y_

            