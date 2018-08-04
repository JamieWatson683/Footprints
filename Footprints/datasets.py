import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import scipy.ndimage as ndi
import os
from skimage import measure, transform


class FootprintsDataset(Dataset):
    def __init__(self, path, augment=False, mask_only=False):
        self.path = path
        self.size = len(os.listdir(path))
        self.augment = augment
        self.mask_only = mask_only

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        datapoint = np.load(self.path+"data_"+str(item)+".npy")
        if self.augment:
            datapoint = self.augment_datapoint(datapoint)
        if self.mask_only:
            inputs = np.expand_dims(datapoint[3], axis=0)
        else:
            inputs = datapoint[0:-1]
        label = datapoint[-1]
        sample = {'inputs': inputs, 'labels': label}
        return sample

    def augment_datapoint(self, datapoint):
        self.image_size = datapoint.shape[1:]
        random_gen = np.random.rand(12)  # alpha, beta, angle, crop(=4), occlude(=5)
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
        # Occlude part of image and mask
        if random_gen[8] > 0.5:
            sprite = int(random_gen[7]*18)
            sprite = np.load("./data/occlusions/person_"+str(sprite)+".npy")
            image, mask = self.occlude_image_and_mask(image, mask, sprite)
        # Flip horizontally
        if random_gen[11] > 0.5:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)
            footprint = np.flip(footprint, axis=1)

        datapoint[0:3] = np.transpose(image, [2,0,1])
        datapoint[3] = mask > 0.5
        datapoint[4] = footprint > 0.5

        return datapoint

    def amend_contrast(self, image, alpha, beta):  # (0.75,1.25 | -0.1,0.1)
        image = image * alpha + beta
        image = np.minimum(image, 1.0)
        image = np.maximum(image, 0.0)
        return image

    def rotate_image(self, image, angle):  # +- 3
        image = ndi.rotate(image, angle, reshape=False, order=1)
        return image

    def crop_and_resize(self, image, topleft, bottomright, binary=False):  # (25,25 | 110,240)
        crop = image[topleft[0]:bottomright[0], topleft[1]:bottomright[1]]
        crop = cv2.resize(crop, (self.image_size[1], self.image_size[0]))
        if binary:
            crop = (crop > 0.5).astype(float)
        return crop

    def occlude_image_and_mask(self, img, mask, sprite):
        mask = (mask > 0) * 1
        bbox = measure.regionprops(mask)[0].bbox
        y = bbox[0]
        x = bbox[1]
        random_nums = np.random.rand(3)
        sprite_y = int(sprite.shape[0] * np.maximum(random_nums[0], 0.2))
        sprite_x = int(sprite.shape[1] * np.maximum(random_nums[0], 0.2))
        sprite = transform.resize(sprite, (sprite_y, sprite_x), preserve_range=True)
        y_max = np.minimum(128, y + sprite_y)
        x_max = np.minimum(256, x + sprite_x)
        img[y:y_max, x:x_max] = np.maximum(
            img[y:y_max, x:x_max] - np.expand_dims(sprite[:y_max - y, :x_max - x, 3], -1), 0)
        img[y:y_max, x:x_max] = img[y:y_max, x:x_max] + sprite[:y_max - y, :x_max - x, :3] / 255 * np.expand_dims(
            sprite[:y_max - y, :x_max - x, 3], -1)
        mask[y:y_max, x:x_max] = np.maximum(mask[y:y_max, x:x_max] - sprite[:y_max - y, :x_max - x, 3], 0)
        return img, mask
