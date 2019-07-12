import os
import sys

sys.path.append('../')

from brats.train import config
from unet3d.prediction import run_validation_cases


def main():
    prediction_dir = os.path.abspath("prediction")
    run_validation_cases(validation_keys_file=config["validation_file"],
                         model_file=config["weigths_file"],
                         training_modalities=config["training_modalities"],
                         labels=config["labels"],
                         hdf5_file=config["data_file"],
                         output_label_map=True,
                         output_dir=prediction_dir, is_weight=True, config=config)


if __name__ == "__main__":
    main()
