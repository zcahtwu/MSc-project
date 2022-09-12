import h5py
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from loader import NewDataset
import torch
import pickle
from loader import Loader
import csv
import pandas as pd
# Change data path
all_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideTestSujit_Axial.h5'
# all_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/Dataset/AbideTestSujitNewNormalization_Axial.h5'
# all_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestSujitNewNormalization_Axial.h5'
# all_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestNewExtractedAxial.h5'
# all_path = r'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/AbideTestNewENAxial.h5'

# Change score path
current = 'Axial'
# current = 'only_intensity'
# current = 'axial_re_extraction'
# current = 'new_intensity_axial'

fc_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/FCsavedscore/fc_testr.pkl'
# fc_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/FCsavedscore/val1.pkl'
dcnn_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/DCNNsavedscore/DCNNscore_test3.pkl'
# dcnn_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/DCNNsavedscore/DCNNscore_val.pkl'


# csv path:
csv_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/csvs/AbideTestNewENAxial.csv'
# csv_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/data_extraction/csvs/AbideValidNewENAxial.csv'

# load scores
with open(dcnn_path, 'rb') as f:
    dcnn_score_list = pickle.load(f)
with open(fc_path, 'rb') as f:
    fc_score = pickle.load(f)
with open(csv_path) as f:
    subject_list = pd.read_csv(f,sep=',',header=None).iloc[:,0]

# pass
# load data
transform_list = transforms.Compose([
                              transforms.ToTensor()
                            ])


_,loader = Loader(all_path,transform_list,32,False,2)

correctted_list = [76,85,105,132,145,147]
for i, data in enumerate(loader, 0):
    if i % 10 ==0:
        print(i)
    batch_image, batch_label = data
    # print(torch.max(batch_image))
    if i in correctted_list:
        batch_label = torch.zeros(32)
    # print(batch_label)
    subject_name = subject_list[i]
    volume_score = round(fc_score[i],3)
    slice_score = dcnn_score_list[32*i:32*(i+1)]
    slice_average = round(np.mean(slice_score),3)
    if batch_label[0] == 1:
        # plot_name = f'./intensity_only/test/1/test{i}.png'
        # plot_name = f'./intensity_only/val/1/val{i}.png'
        plot_name = f'./{current}/test/1/{subject_name}_test{i}.png'
        # plot_name = f'./baseline/val/1/val{i}.png'
    if batch_label[0] == 0:
        # plot_name = f'./intensity_only/test/0/test{i}.png'
        # plot_name = f'./intensity_only/val/0/val{i}.png'
        plot_name = f'./{current}/test/0/{subject_name}_test{i}.png'
        # plot_name = f'./baseline/val/0/val{i}.png'
    fig, axes = plt.subplots(ncols=8, nrows=4,sharex=True, sharey=True,figsize=[24,12])
    fig.suptitle(f'{subject_name}: slice-wise average:{slice_average}, volume score:{volume_score}, label:{batch_label[0]}', fontsize=30)
    fig.subplots_adjust(left = 0.05,right=0.95,bottom=0.1,top = 0.9)
    # plt.tight_layout()
    fig.subplots_adjust(hspace=0.2,wspace = 0.2)
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    for n, ax in enumerate(axes.flatten()):
    # current_slice = batch_image[]
        ax.imshow(batch_image[n,0,:, :], cmap='gray', origin='upper')
        ax.axis('off')   
        ax.set_title(f'{round(slice_score[n],3)}', fontsize=20)
    plt.savefig(plot_name)
    plt.close()