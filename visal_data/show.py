# import packages
import os
import torch
import numpy as np
import pickle

# current = 'Axial'
# current = 'only_intensity'
current = 'new_intensity_axial'
# current = 'axial_re_extraction'
file_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/FCsavedscore/fc_test3.pkl'
file_path = f'/scratch0/tianqiwu/project/UCL_MSc_dsml_project/{current}/DCNNsavedscore/DCNNscore_test3.pkl'

with open(file_path, 'rb') as f:
    baseline = pickle.load(f)
idx = 188


# print(np.around(baseline[idx:idx+10],3))
print(np.flip(np.around(baseline[idx*32:(idx+1)*32],3)))