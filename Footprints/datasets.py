import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class FootprintsDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.size = len(os.listdir(path))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        data_point = np.load(self.path+"data_"+str(item)+".npy")
        inputs = data_point[0:-1]
        label = data_point[-1]
        sample = {'inputs': inputs, 'labels': label}

        return sample