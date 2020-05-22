import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
from PIL import Image

ext = 'nii.gz'

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

def get_slice_of_patient(subject_path, modal_name, slice_idx, plane='axial'):
    subject_name = os.path.basename(subject_path)
    modal_file = '{}_{}.{}'.format(subject_name, modal_name, ext)
    modal_path = os.path.join(subject_path, modal_file)
    image = nib.load(modal_path)
    arr = image.get_data()
    if plane == 'axial':
        slice = arr[:, :, slice_idx]
    elif plane == 'sagittal':
        slice = arr[slice_idx, :, :]
    elif plane == 'coronal':
        slice = arr[:, slice_idx, :]
    else:
        print('Plane is in [axial, sagittal, coronal]')

    return slice

def show_image(rgb_image):
    fig, axes = plt.subplots(1, 1)
    image_to_show = np.rot90(rgb_image, 1, (1, 0))
    axes.imshow(image_to_show, origin='lower')
    axes.axes.xaxis.set_visible(False)
    axes.axes.yaxis.set_visible(False)

def scale_slice_intensity(slice, new_range):
    slice = slice.astype(np.float32)
    new_min_val, new_max_val = new_range
    max_val = np.max(slice)
    min_val = np.min(slice)
    new_range = (new_max_val - new_min_val)
    old_range = (max_val-min_val)
    new_slice = (slice-min_val)*new_range/old_range + new_min_val
    return new_slice

def show_slice(slice):
    fig, axes = plt.subplots(1, 1)
    # for easy view, rotate horizon to vertical
    slice_to_show = np.rot90(slice, 1, (1, 0))
    axes.imshow(slice_to_show, cmap="gray", origin="lower")
    axes.axes.xaxis.set_visible(False)
    axes.axes.yaxis.set_visible(False)

def get_slices_over_modals(subject_path, modal_list, slice_idx_list, output_folder):
    subject_name = os.path.basename(subject_path)

    plane_idx = zip(plane_list, slice_idx_list)
    for plane, idx in plane_idx:
        for modal in modal_list:
            slice = get_slice_of_patient(subject_path, modal, idx, plane)
            show_slice(slice)
            img_name = '{}_{}_{}_{}.png'.format(subject_name, modal, idx, plane)
            output_path = os.path.join(output_folder, subject_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            img_path = os.path.join(output_path, img_name)
            plt.savefig(img_path, bbox_inches='tight')

def overlay_mask(subject_fd, modal_name, mask_path, index, plane, output_fd, mode):
    subject_name = os.path.basename(subject_fd)
    modal_file = '{}_{}.{}'.format(subject_name, modal_name, ext)
    modal_path = os.path.join(subject_fd, modal_file)
    
    image = nib.load(modal_path)
    arr = image.get_data()

    mask_image = nib.load(mask_path)
    mask_arr = mask_image.get_data()

    if plane == 'axial':
        slice = arr[:, :, index]
        mask_slice = mask_arr[:, :, index]
    elif plane == 'sagittal':
        slice = arr[index, :, :]
        mask_slice = mask_arr[index, :, :]
    elif plane == 'coronal':
        slice = arr[:, index, :]
        mask_slice = mask_arr[:, index, :]
    else:
        print('Plane is in [axial, sagittal, coronal]')
    
    scaled_slice = scale_slice_intensity(slice, (0, 255)).astype(np.uint8)
    
    edema = mask_slice == 2
    ncr = mask_slice == 1
    eh = mask_slice == 4
    w, h = slice.shape
    new_slice = np.broadcast_to(scaled_slice, (3, w, h))
    image_arr = np.transpose(new_slice, (1, 2, 0))
    image_arr.setflags(write=1)

    mask_image = np.zeros((w, h, 3), dtype=np.uint8)
    mask_image[edema] = red
    mask_image[ncr] = green
    mask_image[eh] = blue

    image_arr[edema] = 0
    image_arr[ncr] = 0
    image_arr[eh] = 0
    overlay = image_arr + mask_image
    
    output_file = '{}_{}_{}_{}_{}.png'.format(subject_name, modal_name, index, plane, mode)
    output = os.path.join(output_fd, subject_name)
    
    if not os.path.exists(output):
        os.makedirs(output)
    
    output_path = os.path.join(output, output_file)
    show_image(overlay)
    plt.savefig(output_path, bbox_inches='tight')
    

# Get slice of subject  
subject_path = 'E:\machinelearning\data\BRAST2018\Train\HGG\Brats18_TCIA01_147_1'
modal_list = ['t1', 't1ce', 'flair', 't2']
slice_idx_list = [143, 153]
plane_list = ['sagittal', 'coronal']
output_folder = 'E:/luanvantn/images/3dResidualUnet'

# get_slices_over_modals(subject_path, modal_list, slice_idx_list, output_folder)

# Get overlay result
pred_mask_path = "E:/machinelearning/data/BRAST2018/Result/3dResidualUNet/restore_size_prediction/Brats18_TCIA01_147_1/Brats18_TCIA01_147_1_prediction.nii.gz"
label_mask_path = 'E:\machinelearning\data\BRAST2018\Train\HGG\Brats18_TCIA01_147_1\Brats18_TCIA01_147_1_seg.nii.gz'

# Axial
# overlay_mask(subject_path, 't2', pred_mask_path, 74, 'axial', output_folder, 'pred')
# overlay_mask(subject_path, 't2', label_mask_path, 74, 'axial', output_folder, 'label')

#Sagittal
overlay_mask(subject_path, 't2', pred_mask_path, 143, 'sagittal', output_folder, 'pred')
overlay_mask(subject_path, 't2', label_mask_path, 143, 'sagittal', output_folder, 'label')

#Coronal
# overlay_mask(subject_path, 't2', pred_mask_path, 153, 'coronal', output_folder, 'pred')
# overlay_mask(subject_path, 't2', label_mask_path, 153, 'coronal', output_folder, 'label')

# plt.show()