import csv
import numpy as np
import h5py
import torch
from loader import NewDataset,Loader
import nibabel as nib
import pandas as pd
# define training data path
train_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideTrainSujit_Axial.h5'
# define validation data path
val_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideValidSujit_Axial.h5'
# define test data path
test_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideTestSujit_Axial.h5'

# csv_path
train_csv = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/csvs/AbideTrainNewENAxial.csv'
val_csv = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/csvs/AbideValidNewENAxial.csv'
test_csv = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/csvs/AbideTestNewENAxial.csv'


with open(train_csv) as f:
    train_list = pd.read_csv(f,sep=',',header=None).iloc[:,0]
with open(val_csv) as f:
    val_list = pd.read_csv(f,sep=',',header=None).iloc[:,0]
with open(test_csv) as f:
    test_list = pd.read_csv(f,sep=',',header=None).iloc[:,0]

    
batch_size = 32
length_train, trainloader = Loader(train_path,None,batch_size, shuffle = False, num_workers = 8)
length_val, valloader = Loader(val_path,None,batch_size,shuffle = False,num_workers = 8)
length_test, testloader = Loader(test_path,None,batch_size,shuffle = False,num_workers = 8)



def normalize(data,data_path):
    img = nib.load(data_path)
    img_data = img.get_fdata()
    # print(data[0,0,0])
    # print(f'data min = {np.amin(img_data)},data max = {np.amax(img_data)},data 99.9 Q = {np.percentile(img_data,99.9)}')
    img_data = (img_data - np.amin(img_data)) / (np.amax(img_data) - np.amin(img_data))

    # print(f'data min = {np.amin(img_data)},data max = {np.amax(img_data)},data 99.9 Q = {np.percentile(img_data,99.9)}')
    # print(f'slice min= {torch.amin(data)}, slice max= {torch.amax(data)},slice 99.9 Q= {torch.quantile(data,0.999)}')   
    data = (data - np.amin(img_data)) / (np.percentile(img_data,99.9) - np.amin(img_data))

    # data = (data - torch.min(data)) / (torch.quantile(data,0.999) - torch.min(data))
    data[data>1] = 1

    return data

# for i, dataset in enumerate(testloader, 0):
#     # if i % 10 ==0:
#     #     print(i)
#     data_file = f'/cs/research/medic/mri_aqc/abide_organized/5_CorrectedOrientationDataset/niis/{test_list[i]}.nii.gz'
#     print(test_list[i])
#     inputs, labels = dataset
#     # print(labels.shape)
#     # print(inputs.shape)
#     # print(torch.amax(inputs))

#     inputs = normalize(inputs,data_file)
#     break

# --------------------------------------------
# Train:
image = torch.zeros((21056,256,256),dtype = float)
# print(image)
label = torch.zeros(21056, dtype = float)

for i, dataset in enumerate(trainloader, 0):
    if i % 10 ==0:
        print(i)
    data_file = f'/cs/research/medic/mri_aqc/abide_organized/5_CorrectedOrientationDataset/niis/{train_list[i]}.nii.gz'
    inputs, labels = dataset
    # print(labels.shape)
    # print(inputs)
    inputs = normalize(inputs,data_file)

    # print(i)
    # print(inputs)
    image[32*i :32*(i+1),:,:] = inputs
    label[32*i :32*(i+1)] = labels
image = image.to(torch.float)
label = label.to(torch.float)

print(f'{i+1} volume for training')
traindata = '/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTrainSujitNewNormalization_Axial.h5'
hf = h5py.File(traindata, 'w')
hf.create_dataset('image', data=image)
hf.create_dataset('label', data=label)
hf.close()
# # --------------------------------------------
# val:
image = torch.zeros((7040,256,256))
label = torch.zeros(7040)
for i, dataset in enumerate(valloader, 0):
    if i % 10 ==0:
        print(i)
    data_file = f'/cs/research/medic/mri_aqc/abide_organized/5_CorrectedOrientationDataset/niis/{val_list[i]}.nii.gz'
    inputs, labels = dataset
    # print(labels.shape)
    # print(inputs[1])
    inputs = normalize(inputs,data_file)
    # print(inputs[1])
    image[32*i :32*(i+1),:,:] = inputs
    label[32*i :32*(i+1)] = labels
image = image.to(torch.float)
label = label.to(torch.float)
print(f'{i+1} volume for val')
valdata = '/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideValidSujitNewNormalization_Axial.h5'
hf = h5py.File(valdata, 'w')
hf.create_dataset('image', data=image)
hf.create_dataset('label', data=label)
hf.close()

# # --------------------------------------------
# Test:
image = torch.zeros((7072,256,256))
label = torch.zeros(7072)
for i, dataset in enumerate(testloader, 0):
    if i % 10 ==0:
        print(i)
    data_file = f'/cs/research/medic/mri_aqc/abide_organized/5_CorrectedOrientationDataset/niis/{test_list[i]}.nii.gz'
    inputs, labels = dataset
    # print(labels.shape)
    # print(inputs[1])
    inputs = normalize(inputs,data_file)
    # print(inputs[1])
    image[32*i :32*(i+1),:,:] = inputs
    label[32*i :32*(i+1)] = labels
image = image.to(torch.float)
label = label.to(torch.float)
print(f'{i+1} volume for test')
testdata = '/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestSujitNewNormalization_Axial.h5'
hf = h5py.File(testdata, 'w')
hf.create_dataset('image', data=image)
hf.create_dataset('label', data=label)
hf.close()