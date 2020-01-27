import numpy as np
import nibabel as nib
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

def main():
    header = ("WholeTumor", "TumorCore", "EnhancingTumor")
    masking_functions = (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
    rows = list()
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
        rows.append([dice_coefficient(func(truth), func(prediction))for func in masking_functions])

    df = pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
    df.to_csv("./prediction/brats_scores.csv")

    scores = dict()
    for index, score in enumerate(df.columns):
        values = df.values.T[index]
        scores[score] = values[np.isnan(values) == False]

    plt.boxplot(list(scores.values()), labels=list(scores.keys()))
    plt.ylabel("Dice Coefficient")
    plt.savefig("validation_scores_boxplot.png")
    plt.close()

    visualize_training_process('./training.log')

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

if __name__ == "__main__":
    main()
    #visualize_training_process('./training.log')
