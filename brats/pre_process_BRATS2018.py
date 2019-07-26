import sys
sys.path.append('../')

from brats.preprocess import convert_brats_data

convert_brats_data("brats/data/BRAST2018_not_processed/Train", "brats/data/BRATS2018_preprocessed_cont/Train")