import nibabel as nib
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time

from slice_reshape import Reshape

slice_range = [i for i in range(70,90,2)]
print(len(slice_range))
# define data directory
data_path = f'/cs/research/medic/mri_aqc/abide_organized/2_ExtractedImages'
# list all files
dir_list = os.listdir(data_path)[1:]
# loop over all data:
for i in range(len(dir_list)):
    current_data_name = dir_list[i]
    # current_data_name = 'CMU_50661'
    centre = 'NYU'
    if centre not in current_data_name:
        continue
    # define current data
    current_data = os.path.join(data_path, current_data_name, f'anat/NIfTI/mprage.nii.gz')
    # current_data = os.path.join(data_path, current_data_name, f'hires/NIfTI/hires.nii.gz')
    image_id = f'./data_extraction/sagittal/{current_data_name}.png'
    # load data
    try:
        img = nib.load(current_data)
    except:
        continue
    pixel_dim = img.header.get_zooms()[0]
    # breakand
    img_data = img.get_fdata()
    # print(img_data.shape)
    # define nomalization
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    img_data = normalize(img_data)
    
    # Change the shapes
    img_data = Reshape(img_data)
    # check the data after padding have the right shape
    assert img_data.shape == (256,256,256)
    mean_list = []
    std_list = []
    for j in slice_range:
        mean_list.append(np.mean(img_data[j,32:224,32:224]))
        std_list.append(np.std(img_data[j,32:224,32:224]))

    # plt.hist(mean_list,bins = 'auto')
    # plt.savefig(f'{centre} mean')
    # plt.close()

    plt.plot(slice_range,std_list)
    plt.savefig(f'{current_data_name} std')
    plt.close()

    plt.plot(slice_range,mean_list)
    plt.savefig(f'{current_data_name} mean')
    plt.close()
    # show start and end slice and two neighbours
    fig, axes = plt.subplots(ncols=5, nrows=2,figsize=(20,20))
    for n, ax in enumerate(axes.flatten()):
        current_slice = slice_range[n]
        ax.imshow(img_data[current_slice, :, :].T, cmap='gray', origin='lower')
        ax.axis('off')   
        ax.set_title('n=%i, slices=%i' % ((n-1),current_slice), fontsize=20)
    plt.savefig(image_id)
    plt.close()
    # break