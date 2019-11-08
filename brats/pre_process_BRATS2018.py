import sys
import glob
import os
sys.path.append('./')
from unet3d.data import write_data_to_file, open_data_file
from unet3d.utils import pickle_dump


# import pickle

# from brats.preprocess import convert_brats_data

# convert_brats_data("brats/data/BRAST2018_not_processed/Train", "brats/data/BRATS2018_preprocessed_cont/Train")

# def pickle_load(in_file):
#     with open(in_file, "rb") as opened_file:
#         return pickle.load(opened_file)

# filenamevalidx = 'isensee_validation_ids_1.pkl'

# a = pickle_load(filenamevalidx)
# b = a.sort()

# print(a)
# print(len(a))

config = dict()
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["data_file"] = "brats_data_isensee_2018_pieces.h5"
config["image_shape"] = (128, 128, 128)
config["training_file"] = "isensee_training_ids_pieces.pkl"
config["validation_file"] = "isensee_validation_ids_pieces.pkl"

def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname(__file__), "data", "BRATS2018_preprocessed", "Train", "*", "*")):
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in config["all_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files

# training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

# write_data_to_file(training_files, config["data_file"], image_shape=config["image_shape"],
#                            subject_ids=subject_ids)

# data_file_opened = open_data_file(config["data_file"])

# print(data_file_opened.root.data.shape)

train_list = [0, 1, 2, 3, 5, 6, 7, 8]
val_list = [4, 9]

pickle_dump(train_list, config["training_file"])
pickle_dump(val_list, config["validation_file"])
