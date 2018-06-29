import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class FootprintsDataset(Dataset):
    def __init__(self, input_path, label_path):
        self.input_path = input_path
        self.label_path = label_path
        self.size = len(os.listdir(input_path))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        inputs = np.load(self.input_path+"input_"+str(item)+".npy")
        label = np.load(self.label_path+"label_"+str(item)+".npy")
        sample = {'inputs': inputs, 'labels': label}

        return sample