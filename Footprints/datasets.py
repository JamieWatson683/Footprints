import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import scipy.ndimage as ndi
import os


class FootprintsDataset(Dataset):
    def __init__(self, path, augment=False):
        self.path = path
        self.size = len(os.listdir(path))
        self.augment = augment

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
        self.image_size = datapoint.shape[1:]
        random_gen = np.random.rand(7) # alpha, beta, angle, top, left, bottom, right
        datapoint_t = np.transpose(datapoint, [1,2,0])
        image = datapoint_t[:,:,:3]
        mask = datapoint_t[:,:,3]
        footprint = datapoint_t[:,:,4]

        # Amend contrast
        image = self.amend_contrast(image, 0.75+random_gen[0]*0.5, -0.1+random_gen[1]*0.2)
        # Crop and resize
        top = int(random_gen[3]*25)
        left = int(random_gen[4]*25)
        bottom = int(self.image_size[0] - random_gen[5]*18)
        right = int(self.image_size[1] - random_gen[6]*16)
        image = self.crop_and_resize(image, (top,left), (bottom,right))
        mask = self.crop_and_resize(mask, (top, left), (bottom, right), binary=True)
        footprint = self.crop_and_resize(footprint, (top, left), (bottom, right), binary=True)
        # Rotate all
        angle = -3 + random_gen[2] * 6
        image = self.rotate_image(image, angle)
        mask = self.rotate_image(mask, angle)
        footprint = self.rotate_image(footprint, angle)

        datapoint[0:3] = np.transpose(image, [2,0,1])
        datapoint[3] = mask
        datapoint[4] = footprint

        return datapoint

    def amend_contrast(self, image, alpha, beta):  # (0.75,1.25 | -0.1,0.1)
        image = image * alpha + beta
        image = np.minimum(image, 1.0)
        image = np.maximum(image, 0.0)
        return image

    def rotate_image(self, image, angle):  # +- 3
        image = ndi.interpolation.rotate(image, angle, reshape=False)
        return image

    def crop_and_resize(self, image, topleft, bottomright, binary=False):  # (25,25 | 110,240)
        crop = image[topleft[0]:bottomright[0], topleft[1]:bottomright[1]]
        crop = cv2.resize(crop, (self.image_size[1], self.image_size[0]))
        if binary:
            crop = (crop > 0.5).astype(float)
        return crop






