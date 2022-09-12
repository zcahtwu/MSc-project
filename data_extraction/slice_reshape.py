import math
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib

def Reshape(img_data):
    # print(img_data.shape)
    # left and right
    if img_data.shape[0] <=256:
        # print(img_data.shape)
        left_pad = math.ceil((256 - img_data.shape[0])/2)
        right_pad = int((256 - img_data.shape[0])/2)
        img_data = np.pad(img_data,((left_pad,right_pad),(0,0),(0,0)),constant_values=0)
    else:
        # print(img_data.shape)
        left_start = math.ceil((img_data.shape[0]-256)/2)
        right_end = img_data.shape[0] - int((img_data.shape[0]-256)/2)
        img_data = img_data[left_start:right_end,:,:]

    # front and back
    if img_data.shape[1] <=256:
        # print(img_data.shape)
        front_pad = math.ceil((256 - img_data.shape[1])/2)
        back_pad = int((256 - img_data.shape[1])/2)
        img_data = np.pad(img_data,((0,0),(front_pad,back_pad),(0,0)),constant_values=0)
    else:
        # print(img_data.shape)
        front_start = math.ceil((img_data.shape[1]-256)/2)
        back_end = img_data.shape[1] - int((img_data.shape[1]-256)/2)
        img_data = img_data[:,front_start:back_end,:]
    
    # up and bottom
    if img_data.shape[2] <=256:
        # print(img_data.shape)
        up_pad = math.ceil((256 - img_data.shape[2])/2)
        bottom_pad = int((256 - img_data.shape[2])/2)
        img_data = np.pad(img_data,((0,0),(0,0),(up_pad,bottom_pad)),constant_values=0)
    else:
        # print(img_data.shape)
        up_start = math.ceil((img_data.shape[2]-256)/2)
        bottom_end = img_data.shape[2] - int((img_data.shape[2]-256)/2)
        img_data = img_data[:,:,up_start:bottom_end]

    # print(img_data.shape)
    return img_data

# # define data directory
# data_path = f'/cs/research/medic/mri_aqc/abide_organized/2_ExtractedImages'
# # list all files
# dir_list = os.listdir(data_path)[1:]
# # loop over all data:
# # for i in range(len(dir_list)):
# # for i in range(600,1000):
# current_data_name = 'Trinity_50237'
# # current_data_name = dir_list[i+1]
# current_data = os.path.join(data_path, current_data_name, f'anat/NIfTI/mprage.nii.gz')

# try:
#     img1 = nib.load(current_data)
# except:
#     pass

# img = img1.get_fdata()
# # print(img_data.shape)

# # define nomalization
# def normalize(data):
#     return (data - np.min(data)) / (np.max(data) - np.min(data))
# img = normalize(img)

# # print(f'{i}th point, {current_data_name}')
# img_data = Reshape(img)

# # visualize the data
# # show +-2 slices
# fig, axes = plt.subplots(ncols=8, nrows=8,figsize=(20,20))  
# # plt.subplots_adjust(left=0.1,
# #             bottom=0.1, 
# #             right=0.5, 
# #             top=0.5, 
# #             wspace=0.3, 
# #             hspace=0.3)

# for n, ax in enumerate(axes.flatten()):
#     # ax.imshow(img_data[0 + n*4, :, :].T, cmap='gray', origin='lower')
#     # ax.imshow(img_data[:, 4*n, :].T, cmap='gray', origin='lower')
#     ax.imshow(img_data[:, :, n*4].T, cmap='gray', origin='lower')
#     ax.axis('off')   
#     # ax.set_title('n=%i, s=%i,e=%i' % ((n-1),start,(start + sum(gap_size[2:n]))), fontsize=10)
# plt.savefig('1')
# plt.close()
# # print(end_time-start_time)