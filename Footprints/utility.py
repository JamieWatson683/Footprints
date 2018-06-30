import numpy as np
from skimage import io, transform, measure
from torch.utils.data import Dataset, DataLoader
import os


class DataProcessor(object):
    """ Class to process raw images, masks and footprints to generate and save network inputs and labels as .npy files.
    Saves data as [C x H x W] as per Pytorch ordering.
        -> C = [R, G, B, Mask, Footprint]
    """
    def __init__(self):
        self.masks = np.array([])
        self.footprints = np.array([])
        self.images = np.array([])
        self.train_data = None
        self.val_data = None
        self.filenames = None

    def load_raw_data(self, path, image_size=(288,512)):
        filenames = os.listdir(path+"ground_masks/masks/")
        filenumber = len(filenames)

        masks = np.zeros([filenumber, image_size[0], image_size[1]], dtype=int)
        footprints = np.zeros_like(masks)
        images = np.zeros([filenumber, image_size[0], image_size[1], 3]) # number of files x h x w x RGB

        index = 0
        for name in filenames:
            mask = np.load(path+"ground_masks/masks/"+name)
            footprint = np.load(path+"ground_footprints/footprints/"+name)
            image = io.imread(path+"/ground_frames/"+name[:-4]+".jpg") / 255

            masks[index] = mask
            footprints[index] = footprint
            images[index] = image

            index += 1

        # Save file name ordering for reconstruction later
        self.filenames = np.array([filenames])

        # if len(self.masks)!=0:  # if data in train_masks then add to it
        #     self.masks = np.concatenate((self.masks, masks), axis=0)
        #     self.footprints = np.concatenate((self.footprints, footprints), axis=0)
        #     self.images = np.concatenate((self.images, images), axis=0)
        # else:  # otherwise set explicitly
        self.masks = masks
        self.footprints = footprints
        self.images = images

    def shuffle_data(self):
        ordering = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[ordering]

    def valitdation_split(self, val_fraction):
        self.shuffle_data()
        self.val_data = self.train_data[:int(round(val_fraction * len(self.train_data)))]
        self.train_data = self.train_data[int(round(val_fraction * len(self.train_data))):]

    def prepare_data(self, output_shape):

        size = len(self.masks)
        # Initialise
        masks = np.zeros([size, output_shape[0], output_shape[1]], dtype=int)
        footprints = np.zeros([size, output_shape[0], output_shape[1]], dtype=float)
        images = np.zeros([size, output_shape[0], output_shape[1], 3])
        for i in range(size):
            # Loop and resize to output_shape
            masks[i] = transform.resize(self.masks[i], output_shape=output_shape, preserve_range=True)
            footprints[i] = transform.resize(self.footprints[i], output_shape=output_shape, preserve_range=True)
            images[i] = transform.resize(self.images[i], output_shape=output_shape, preserve_range=True)

        masks = np.expand_dims(masks, -1)
        footprints = np.expand_dims((footprints > 0.5).astype(int), -1)
        self.train_data = np.concatenate((images, masks, footprints), axis=-1)
        self.train_data = np.transpose(self.train_data, [0,3,1,2]) # for pytorch style inputs (batch x C x H x W)

    def save_data(self, path, val_path=None):
        index = len(os.listdir(path))
        for i in range(len(self.train_data)):
            np.save(path + "data_" + str(i+index), self.train_data[i])

        if val_path:
            index = len(os.listdir(val_path))
            for i in range(len(self.val_data)):
                np.save(val_path + "data_" + str(i + index), self.val_data[i])
