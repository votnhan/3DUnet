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


def get_edema_mask(data):
    return data == 2


def get_necrotic_mask(data):
    return data == 1


def get_non_tumor_mask(data):
    return data == 0
    

header_masking_dict = {
    'over_class': {
        'headers': ('None Tumor', 'Necrotic', 'Edema', 'Enhancing Tumor'),
        'masking_funcs': (get_non_tumor_mask, get_necrotic_mask, get_edema_mask, get_enhancing_tumor_mask)
        },
    'brats_sub_region': {
        'headers': ('Whole Tumor', 'Tumor Core', 'Enhancing Tumor'),
        'masking_funcs': (get_whole_tumor_mask, get_tumor_core_mask, get_enhancing_tumor_mask)
        }
}


def dice_coefficient(truth, prediction):
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))

# Recall positive
def get_Sensitivity(truth, prediction):
  tp = np.sum(truth*prediction)
  return tp / (np.sum(truth))

# Recall negative
def get_Specificity(truth, prediction):
  n_truth = truth == 0
  n_prediction = prediction == 0
  tn = np.sum(n_truth * n_prediction)
  return tn / np.sum(n_truth)


# Precision positive
def get_Precision_positive(truth, prediction):
    tp = np.sum(truth*prediction)
    return tp / (np.sum(prediction))


# Precision negative
def get_Precision_negative(truth, prediction):
    n_truth = truth == 0
    n_prediction = prediction == 0
    tn = np.sum(n_truth * n_prediction)
    return tn / np.sum(n_prediction)


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
    'precision_positive': get_Precision_positive,
    'precision_negative': get_Precision_negative,
    'hd_distance': get_Hd_distance
}

y_labels_dict = {
    'dice_score': 'Dice Score',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'precision_positive': 'Precision positive',
    'precision_negative': 'Precision negative',
    'hd_distance': 'Hausdorff Distance'
}

def get_file_path_for_evaluation(mode, output_path, label_path):
    list_outputs = []
    list_labels = []
    pattern = os.path.join(output_path, '*')
    subject_paths = glob.glob(pattern)
    if mode == 'model_mode':
        assert output_path == label_path
        for subject_path in subject_paths:
            output = os.path.join(subject_path, 'prediction.nii.gz')
            label = os.path.join(subject_path, 'truth.nii.gz')
            list_outputs.append(output)
            list_labels.append(label)

    elif mode == 'original_mode':
        pattern_label = os.path.join(label_path, '*', '*')
        label_paths = glob.glob(pattern_label)
        for subject_path in subject_paths:
            subject_name = os.path.basename(subject_path)
            output_file = '{}_prediction.nii.gz'.format(subject_name)
            label_file = '{}_seg.nii.gz'.format(subject_name)
            lbl_container = None
            for lbl in label_paths:
                if subject_name in lbl:
                    lbl_container = lbl
                    break

            output = os.path.join(subject_path, output_file)
            label = os.path.join(lbl_container, label_file)
            list_outputs.append(output)
            list_labels.append(label)

    else:
        print('{} is not implemented !'.format(mode))
    
    return list_outputs, list_labels

def main(metric_names, prediction_path, label_path, output_folder, mode, region_eval='brats_sub_region'):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    headers = header_masking_dict[region_eval]['headers']
    masking_functions = header_masking_dict[region_eval]['masking_funcs']
    rows = [list() for i in range(len(metric_names))]
    subject_ids = list()
    list_outputs, list_labels = get_file_path_for_evaluation(mode, prediction_path, label_path)
    outputs_labels = zip(list_outputs, list_labels)
    for output_path, label_path in outputs_labels:
        if not os.path.exists(output_path):
            continue
        
        subject_id = output_path.split('/')[-2]
        subject_ids.append(subject_id)
        truth_image = nib.load(output_path)
        truth = truth_image.get_data()
        prediction_image = nib.load(label_path)
        prediction = prediction_image.get_data()
        for i, x in enumerate(metric_names):
            if x not in metrics_dict:
                print('{} not in metrics dictionary !'.format(x))
                continue
            metric_val = [metrics_dict[x](func(truth), func(prediction)) for func in masking_functions]
            rows[i].append(metric_val)

    export_csv_files(rows, headers, metric_names, subject_ids, output_folder)
    export_boxplot(rows, headers, metric_names, output_folder)


def export_csv_files(data, header, metric_names, subject_ids, output_folder):
    for i, x in enumerate(metric_names):
        df = pd.DataFrame.from_records(data[i], columns=header, index=subject_ids)
        output_path = os.path.join(output_folder, '{}.csv'.format(x))
        df.to_csv(output_path)
        print('Export csv file for {} metric'.format(x))


def export_boxplot(data, header, metric_names, output_folder):        
    data_arr = np.asarray(data)    
    for i, metric in enumerate(metric_names):
        scores = dict()
        for j, score in enumerate(header):
            values = data_arr[i, :, j]
            scores[score] = values[np.isnan(values) == False]
        
        plt.boxplot(list(scores.values()), labels=list(scores.keys()))
        plt.ylabel(y_labels_dict[metric])
        plt.savefig(os.path.join(output_folder, '{}_boxplot.png'.format(metric)))
        plt.close()


def visualize_training_process(logfile, output_name):
    if os.path.exists(logfile):
        training_df = pd.read_csv(logfile).set_index('epoch')
        plt.plot(training_df['loss'].values, label='training loss')
        plt.plot(training_df['val_loss'].values, label='validation loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.xlim((0, len(training_df.index)))
        plt.legend(loc='upper right')
        plt.savefig('{}.png'.format(output_name))
        return 'Done'

    return 'Log File Not Found'

parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--metrics_name', type=str, nargs='+', 
            default=['dice_score', 'sensitivity', 'specificity', 'hd_distance'])

parser.add_argument('--output_folder', type=str, default='output')
parser.add_argument('--prediction_folder', type=str, default='prediction')
parser.add_argument('--label_folder', type=str, default='prediction')
parser.add_argument('--mode', type=str, default='model_mode', help='model_mode or original_mode')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.metrics_name, args.prediction_folder, args.label_folder, 
    args.output_folder, args.mode)
    #visualize_training_process('./training.log')
