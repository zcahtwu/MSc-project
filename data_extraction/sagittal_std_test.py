from platform import java_ver
import nibabel as nib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time

from slice_reshape import Reshape
# define data directory
data_path = f'/cs/research/medic/mri_aqc/abide_organized/2_ExtractedImages'
# list all files
dir_list = os.listdir(data_path)[1:]
# loop over all data:
for i in range(len(dir_list)):
# for i in range(780,790):

    current_data_name = dir_list[i]
    centre = 'KKI'
    if centre not in current_data_name:
        continue
    # current_data_name = 'NYU_51087'
    # define current data
    current_data = os.path.join(data_path, current_data_name, f'anat/NIfTI/mprage.nii.gz')
    # current_data = os.path.join(data_path, current_data_name, f'hires/NIfTI/hires.nii.gz')
    image_id = f'./data_extraction/sagittal/{current_data_name}.png'
    # load data
    try:
        img = nib.load(current_data)
    except:
        continue
    # print(img.header)
    # break
    img_data = img.get_fdata()
    # define nomalization
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    img_data = normalize(img_data)
    
    # Change the shapes
    img_data = Reshape(img_data)
    # check the data after padding have the right shape
    assert img_data.shape == (256,256,256)
    square_std_list = []
    # define the starting slice
    threshold_mean = 0.02
    threshold_std = 0.05
    for start in range(32,82,2):
        # square_mean_list = []

        n_non_blank = 0
        for row in range(6):
            for col in range(6):
                square_mean = np.mean(img_data[start,(start+32*row):(start+32*(row+1)),(start+32*col):(start+32*(col+1))])
                square_std = np.std(img_data[start,(start+32*row):(start+32*(row+1)),(start+32*col):(start+32*(col+1))])
                if not(square_mean < threshold_mean):# or square_std < threshold_std):
                    square_std_list.append(square_std)
                    n_non_blank +=1
        # print(max(square_std_list))
        if n_non_blank >= 24:
            break
    start_slice = start
    # # find the end slice
    # for end in range (160,200,2):
    #     # square_mean_list = []
    #     # square_std_list = []
    #     n_non_blank = 0
    #     for row in range(6):
    #         for col in range(6):
    #             square_mean = np.mean(img_data[end,(start+32*row):(start+32*(row+1)),(start+32*col):(start+32*(col+1))])
    #             square_std = np.std(img_data[end,(start+32*row):(start+32*(row+1)),(start+32*col):(start+32*(col+1))])
    #             if not( square_mean < threshold_mean and square_std < threshold_std):
    #                 n_non_blank +=1
    #     if n_non_blank < 24:
    #         break
    # end_slice = end
    # print(f'{current_data_name},start slice:{start_slice}, end slice:{end_slice}')
    # # The total useful length of the image in this direction
    # brain_length = end_slice - start_slice
    # # print(f'brain_length = {brain_length}')
    # # number of gaps
    # n_gaps = 31
    # # define gap_size
    # gap_size = (end_slice - start_slice)/n_gaps

# # ---------------------------------------------------------------------------
    # # # show start and end slice and two neighbours
    # fig, axes = plt.subplots(ncols=3, nrows=2)
    # for n, ax in enumerate(axes.flatten()):
    #     if n - 2 <=0:
    #         current_slice = start_slice + (n-1)*4
    #     else:
    #         current_slice = end_slice + (n-4) *4
    #     ax.imshow(img_data[current_slice, :, :].T, cmap='gray', origin='lower')
    #     ax.axis('off')   
    #     ax.set_title('n=%i, slices=%i' % ((n-1),current_slice), fontsize=20)
    # plt.savefig(image_id)
    # plt.close()

bin_list = [i/100 for i in range(20)]
# print(square_std_list)
plt.hist(square_std_list,bins = bin_list)
plt.savefig(centre)