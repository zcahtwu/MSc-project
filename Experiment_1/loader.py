# import library
import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

# define dataset to use data loader
class NewDataset(Dataset):
    def __init__(self, data_path,transform=None):
        self.length = len(h5py.File(data_path, 'r')['label'])
        self.path = data_path
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = h5py.File(self.path, 'r')['image'][idx]
        label = h5py.File(self.path, 'r')['label'][idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# define data loader
# define seed work to fix in data loader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
def Loader(data_path, transform,batch_size, shuffle, num_workers = 0):
    dataset = NewDataset(data_path,transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers = num_workers,worker_init_fn=seed_worker,generator=g)
    return dataset.length, loader