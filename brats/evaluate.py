import numpy as np
import nibabel as nib
import argparse
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


def get_whole_tumor_mask(data):
    return data > 0


def get_tumor_core_mask(data):
    return np.logical_or(data == 1, data == 4)


def get_enhancing_tumor_mask(data):
    return data == 4


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

# TPR

def get_Sensitivity(truth, prediction):
  tp = np.sum(truth*prediction)
  return tp / (np.sum(truth))

# TNR
def get_Specificity(truth, prediction):
  n_truth = truth == 0
  n_prediction = prediction == 0
  tn = np.sum(n_truth * n_prediction)
  return tn / np.sum(n_truth)

# Hausdorff distance
def get_Hd_distance(truth, prediction):
  coor_truth = np.where(truth==1)
  coor_truth = np.asarray(coor_truth).T
  coor_preds = np.where(prediction==1)
  coor_preds = np.asarray(coor_preds).T
  d_ab = directed_hausdorff(coor_truth, coor_preds)[0]
  d_ba = directed_hausdorff(coor_preds, coor_truth)[0]
  hd_ab = max(d_ab, d_ba)
  return hd_ab

metrics_dict = {
    'dice_score': dice_coefficient,
    'sensitivity': get_Sensitivity,
    'specificity': get_Specificity,
    'hd_distance': get_Hd_distance
}

y_labels_dict = {
    'dice_score': 'Dice Score',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'hd_distance': 'Hausdorff Distance'
}

def main(metric_names):
    header = ("Whole Tumor", "Tumor Core", "Enhancing Tumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()*len(metric_names)
    subject_ids = list()
    for case_folder in glob.glob("prediction/*"):
        if not os.path.isdir(case_folder):
            continue
        subject_ids.append(os.path.basename(case_folder))
        truth_file = os.path.join(case_folder, "truth.nii.gz")
        truth_image = nib.load(truth_file)
        truth = truth_image.get_data()
        prediction_file = os.path.join(case_folder, "prediction.nii.gz")
        prediction_image = nib.load(prediction_file)
        prediction = prediction_image.get_data()
        for i, x in enumerate(metric_names):
            if x not in metrics_dict:
                print('{} not in metrics dictionary !'.format(x))
                continue
            metric_val = [metrics_dict[x](func(truth), func(prediction)) for func in masking_functions]
            rows[i].append(metric_val)

    for i, x in enumerate(metric_names):
        df = pd.DataFrame.from_records(rows[i], columns=header, index=subject_ids)
        df.to_csv("./prediction/{}.csv".format(x))

        scores = dict()
        for index, score in enumerate(df.columns):
            values = df.values.T[index]
            scores[score] = values[np.isnan(values) == False]

        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel(y_labels_dict[x])
        plt.savefig("{}_boxplot.png".format(x))
        plt.close()

def visualize_training_process(logfile):
    if os.path.exists(logfile):
        training_df = pd.read_csv(logfile).set_index('epoch')
        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('loss_graph.png')
        return 'Done'

    return 'Log File Not Found'

parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--metrics_name', type=str, nargs='+', 
            default=['dice_score', 'sensitivity', 'specificity', 'hd_distance'])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.metrics_name)
    #visualize_training_process('./training.log')
