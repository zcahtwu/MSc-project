import nibabel as nib
import nibabel.processing as processing
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import math
import time
from slice_reshape import Reshape
from axial_extract_function import axial_extraction
# define data directory
# data_path = f'/cs/research/medic/mri_aqc/abide_organized/2_ExtractedImages'
data_path = f'/cs/research/medic/mri_aqc/abide_organized/5_CorrectedOrientationDataset/niis'
# list all files
dir_list = os.listdir(data_path)[:]
# loop over all data:
for i in range(len(dir_list)):
# for i in range(780,790):

    current_data_name = dir_list[i]
    # print (current_data_name)
    # break
    # current_data_name = 'UCLA_51213'
    # centre = 'Trinity_50246'
    # centre = 'NYU_50985'
    centre = 'Yale_50611'
    # centre = 'Caltech_51459'
    # centre = 'USM_50530'
    if centre not in current_data_name:
        continue
    # define current data
    # current_data = os.path.join(data_path, current_data_name, f'anat/NIfTI/mprage.nii.gz')
    current_data = os.path.join(data_path,current_data_name)
    image_id = f'./axial/{current_data_name}.png'
    # load data
    try:
        img = nib.load(current_data)
        # print(current_data_name)
        print(img.shape)
    except:
        continue
    # img_data, ls = axial_extraction(current_data_name,img)
    # print(ls)
    # print(img.shape)
    # if 'USM' in current_data_name:
        # print(img.shape)
    #     voxel_size = [img.header.get_zooms()[0],1.0,1.0]
    #     img = processing.resample_to_output(img, voxel_size)
    #     # print(img.shape)
    
    pixel_dim = img.header.get_zooms()[2]
    img_data = img.get_fdata()

    def normalize1(data):
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # print(np.percentile(data,99.9))
        # data[data>1] = 1
        return data
    def normalize2(data):
        # print('2:======')
        # print(np.amax(data))
        # print(np.percentile(data,99.9))
        data = (data - np.min(data)) / (np.percentile(data,99.9) - np.min(data))
        data[data>1] = 1
        return data
    img_data1 = normalize1(img_data)
    img_data2 = normalize2(img_data).flatten()
    img_data = img_data.flatten()
    
    # print(img_data.shape)
    plt.figure(figsize=(10,5))
    n, bins, patches = plt.hist(img_data, 100, density=True,facecolor='skyblue', alpha=0.75,edgecolor = 'k')
    plt.xlabel('Intensity',fontsize=15)
    plt.ylabel('Prabaility',fontsize=15)
    plt.title('Histagram of the intensity',fontsize=15)
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.xlim(0, np.max(img_data))
    plt.ylim(0, 0.01)
    plt.grid(True)
    plt.savefig('hist_Yale')
    plt.close()
    break




    # print(np.max(img_data))
    # Change the shapes
    # img_data = Reshape(img_data)
    # # check the data after padding have the right shape
    # assert img_data.shape == (256,256,256)

    # define the starting slice
    # threshold_mean = 0.2
    # threshold_std = 0.1
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

    # for start in reversed(range(144,242,2)):
    #     square_std = np.std(img_data[32:224, 32:224, start])
    #     if square_std >= threshold_std:
    #         # print(start)
    #         break
    # start_slice = start - round(10/pixel_dim)

    # # find the end slice
    # end_slice = start_slice - round(120/pixel_dim)
    # # print(np.std(img_data[:,:,end_slice]))
    # while np.std(img_data[:,:,end_slice])<=threshold_std:
    #     end_slice +=2
    #     # start_slice +=2
    # # if start_slice <= 150:
    # #     start_slice = 150
    # #     end_slice = start_slice - round(120/pixel_dim)
    # # if start_slice > 180:
    # #     start_slice = 180
    # #     end_slice = start_slice - round(120/pixel_dim)
    # print(f'{current_data_name},start slice:{start_slice}, end slice:{end_slice}')
    # # # The total useful length of the image in this direction
    # # brain_length = end_slice - start_slice
    # # # print(f'brain_length = {brain_length}')
    # # number of gaps
    # n_gaps = 31
    # # define gap_size
    # gap_size = (start_slice - end_slice) * pixel_dim/n_gaps

# # ---------------------------------------------------------------------------

    # fig, axes = plt.subplots(ncols=8, nrows=4,figsize=(40,20),facecolor = (0,0,0))
    # fig.subplots_adjust(hspace=0,wspace = 0)
    # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # for n, ax in enumerate(axes.flatten()):
    #     # current_slice = start_slice - round(n*gap_size/pixel_dim)
    #     current_slice = ls[31-n]
    #     ax.imshow(np.rot90(img_data[:, :,current_slice],-1), cmap='gray', origin='lower')
    #     ax.axis('off')   
    #     # ax.set_title('n=%i, slices=%i' % ((n+1),current_slice), fontsize=20)
    # plt.savefig(image_id)
    # plt.close()
    # # break