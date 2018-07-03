import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
import os


class FootprintsDataset(Dataset):
    def __init__(self, path, augment=False, crop_pixels=(10, 15), rotation_max=5):
        self.path = path
        self.size = len(os.listdir(path))
        self.augment = augment
        if self.augment:
            self.toPIL = transforms.ToPILImage()
            self.toTensor = transforms.ToTensor()
            self.height_crop = crop_pixels[0]
            self.width_crop = crop_pixels[1]
            self.rotation_max = rotation_max


    def __len__(self):
        return self.size

    def __getitem__(self, item):
        datapoint = np.load(self.path+"data_"+str(item)+".npy")
        if self.augment:
            datapoint = self.augment_datapoint(datapoint)
        inputs = datapoint[0:-1]
        label = datapoint[-1]
        sample = {'inputs': inputs, 'labels': label}
        return sample

    def augment_datapoint(self, datapoint):
        datapoint = (datapoint * 255).astype(np.uint8)
        img = self.toPIL(datapoint[0:3])
        mask = self.toPIL(datapoint[3])
        footprint = self.toPIL(datapoint[-1])
        size = mask.shape

        # Adjust colours
        img = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)

        # Random cropping
        crop_fractions = np.random.rand(4)
        top = crop_fractions[0] * self.height_crop
        bottom = size[0] - 1 - int(crop_fractions[1] * self.height_crop)
        left = crop_fractions[1] * int(crop_fractions[2] * self.width_crop)
        right = size[1] - 1 - int(crop_fractions[3] * self.width_crop)
        img = img.crop(left, top, right, bottom)
        mask = mask.crop(left, top, right, bottom)
        footprint = footprint.crop(left, top, right, bottom)

        # Random flip
        p = np.random.rand()
        if p > 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)
            footprint = F.hflip(footprint)

        # Random rotation
        angle = - self.rotation_max + self.rotation_max * np.random.rand() * 2
        img = img.rotate(angle)
        mask = mask.rotate(angle)
        footprint = footprint.rotate(angle)

        img = self.toTensor(img)
        mask = self.toTensor(mask)
        footprint = self.toTensor(footprint)
        datapoint[0:3] = img.float() / 255
        datapoint[3] = mask.float() / 255
        datapoint[-1] = footprint.float() / 255
        return datapoint







