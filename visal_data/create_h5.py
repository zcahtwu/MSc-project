import numpy as np
import h5py
import torch
from loader import NewDataset,Loader

image = torch.full((32, 256, 256), 0.0299)
# print(image)
label = torch.full((32,), 0)

image = image.to(torch.float)
label = label.to(torch.float)


traindata = '0299.h5'
hf = h5py.File(traindata, 'w')
hf.create_dataset('image', data=image)
hf.create_dataset('label', data=label)
hf.close()
