import os
import sys
import json

sys.path.append('../')

from unet3d.prediction import run_validation_cases

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
