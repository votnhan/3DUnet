import sys
sys.path.append('./')

import pickle

# from brats.preprocess import convert_brats_data

# convert_brats_data("brats/data/BRAST2018_not_processed/Train", "brats/data/BRATS2018_preprocessed_cont/Train")

def pickle_load(in_file):
    with open(in_file, "rb") as opened_file:
        return pickle.load(opened_file)

filenamevalidx = 'isensee_validation_ids_1.pkl'

a = pickle_load(filenamevalidx)
b = a.sort()

print(a)
print(len(a))