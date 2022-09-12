import nibabel as nib
import nibabel.processing as processing
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time
from slice_reshape import Reshape

def axial_extraction(data_name,image):
    current_data_name = data_name
    img = image

    if 'USM' in current_data_name:
        voxel_size = [img.header.get_zooms()[0],1.0,1.0]
        img = processing.resample_to_output(img, voxel_size)
        # print(img.shape)

    pixel_dim = img.header.get_zooms()[2]
    # print(pixel_dim)
    # breakand
    img_data = img.get_fdata()

    def normalize1(data):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # data[data>1] = 1
        return data
    def normalize2(data):
        data = (data - np.min(data)) / (np.percentile(data,99.9) - np.min(data))
        data[data>1] = 1
        return data
    # img_data1 = normalize1(img_data)
    img_data2 = normalize2(img_data)

    # print(np.max(img_data))
    # Change the shapes
    # img_data1 = Reshape(img_data1)
    img_data2 = Reshape(img_data2)

    # check the data after padding have the right shape
    assert img_data2.shape == (256,256,256)

    # define the starting slice
    threshold_std = 0.1

    for start in reversed(range(144,242,2)):
        square_std = np.std(img_data2[32:224, 32:224, start])
        if square_std >= threshold_std:
            # print(start)
            break
    start_slice = start - round(10/pixel_dim)

    # find the end slice
    end_slice = start_slice - round(125/pixel_dim)
    while np.std(img_data2[:,:,end_slice])<=threshold_std:
        end_slice +=2
    # print(f'{current_data_name},start slice:{start_slice}, end slice:{end_slice}')

    # number of gaps
    n_gaps = 31
    # define gap_size
    gap_size = (start_slice - end_slice) * pixel_dim/n_gaps

    idx_list = np.zeros(32)
    for slice in range(len(idx_list)):
        idx_list[slice] = start_slice - round(slice * gap_size / pixel_dim)
    
    return img_data2, idx_list.astype(int).tolist()