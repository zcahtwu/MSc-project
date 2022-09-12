import nibabel as nib
import nibabel.processing as processing
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time
from slice_reshape import Reshape
# define data directory
# data_path = f'/cs/research/medic/mri_aqc/abide_organized/2_ExtractedImages'
data_path = f'/cs/research/medic/mri_aqc/abide_organized/4_2_1_correctOrientation'
# list all files
dir_list = os.listdir(data_path)[1:]
# loop over all data:
for i in range(len(dir_list)):
# for i in range(780,790):

    current_data_name = dir_list[i]
    # current_data_name = 'UCLA_51213'
    centre = 'USM'
    if centre not in current_data_name:
        continue
    # define current data
    # current_data = os.path.join(data_path, current_data_name, f'anat/NIfTI/mprage.nii.gz')
    current_data = os.path.join(data_path,current_data_name)
    image_id = f'./coronal/{current_data_name}.png'
    # load data
    try:
        img = nib.load(current_data)
    except:
        continue
    # print(img.shape)
    pixel_dim = img.header.get_zooms()[1]
    print(pixel_dim)
    if 'USM' in current_data_name:
        voxel_size = [img.header.get_zooms()[0],1.0,1.0]
        img = processing.resample_to_output(img, voxel_size)
        # print(img.shape)
    
    pixel_dim = img.header.get_zooms()[1]
    # print(pixel_dim)
    # breakand
    img_data = img.get_fdata()

    def normalize(data):
        data = (data - np.min(data)) / (np.percentile(data,99.9) - np.min(data))
        data[data>1] = 1
        return data
    img_data = normalize(img_data)
    # print(np.max(img_data))
    # Change the shapes
    img_data = Reshape(img_data)
    # check the data after padding have the right shape
    assert img_data.shape == (256,256,256)

    # define the starting slice
    threshold_mean = 0.2
    threshold_std = 0.1
    # for start in reversed(range(144,242,2)):
    #     # square_mean_list = []
    #     # square_std_list = []
    #     n_non_blank = 0
    #     for row in range(6):
    #         for col in range(6):
    #             square_mean = np.mean(img_data[(32*(row+1)):(32*(row+2)),(32*(col)+1):(32*(col+2)),start])
    #             square_std = np.std(img_data[(32*(row+1)):(32*(row+2)),(32*(col+1)):(32*(col+2)),start])
    #             if not (square_mean < threshold_mean and square_std < threshold_std):
    #                 # square_std_list.append(square_std)
    #                 # print(square_mean)
    #                 n_non_blank +=1
    #     if n_non_blank >= 12:
    #         # print(square_mean)
    #         break

    for start in range(16,80,2):
        square_std = np.std(img_data[32:224 ,start, 32:224])
        if square_std >= threshold_std:
            # print(start)
            break
    start_slice = start + round(5/pixel_dim)

    # find the end slice
    end_slice = start_slice + round(155/pixel_dim)

    print(f'{current_data_name},start slice:{start_slice}, end slice:{end_slice}')
    # # The total useful length of the image in this direction
    # brain_length = end_slice - start_slice
    # # print(f'brain_length = {brain_length}')
    # number of gaps
    n_gaps = 31
    # define gap_size
    gap_size = (start_slice - end_slice) * pixel_dim/n_gaps

# ---------------------------------------------------------------------------
    # show start and end slice and two neighbours
    fig, axes = plt.subplots(ncols=4, nrows=8,figsize=(20,20))
    for n, ax in enumerate(axes.flatten()):
        current_slice = start_slice - round(n*gap_size/pixel_dim)
        # current_slice = 50 + n *5
        ax.imshow(img_data[:,current_slice, :].T, cmap='gray', origin='lower')
        ax.axis('off')   
        ax.set_title('n=%i, slices=%i' % ((n+1),current_slice), fontsize=20)
    plt.savefig(image_id)
    plt.close()
    break
