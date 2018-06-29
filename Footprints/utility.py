import numpy as np
from skimage import io, transform, measure
from torch.utils.data import Dataset, DataLoader
import os


class DataProcessor(object):
    """ Class to process raw images, masks and footprints to generate and save network inputs and labels as .npy files
    """
    def __init__(self):
        self.masks = None
        self.footprints = None
        self.images = None
        self.inputs = None
        self.labels = None
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

        if self.masks:  # if data in train_masks then add to it
            self.masks = np.concatenate((self.masks, masks), axis=0)
            self.footprints = np.concatenate((self.footprints, footprints), axis=0)
            self.images = np.concatenate((self.images, images), axis=0)
        else:  # otherwise set explicitly
            self.masks = masks
            self.footprints = footprints
            self.images = images

    def shuffle_data(self):
        ordering = np.random.permutation(len(self.inputs))
        self.inputs = self.inputs[ordering]
        self.labels = self.labels[ordering]

    def prepare_data(self, output_shape):

        size = len(self.masks)
        # Initialise
        masks = np.zeros([size, output_shape[0], output_shape[1]], dtype=int)
        footprints = np.zeros_like(masks)
        images = np.zeros([size, output_shape[0], output_shape[1], 3])
        for i in range(size):
            # Loop and resize to output_shape
            masks[i] = transform.resize(self.masks[i], output_shape=output_shape)
            footprints[i] = transform.resize(self.footprints[i], output_shape=output_shape)
            images[i] = transform.resize(self.images[i], output_shape=output_shape)

        masks = np.expand_dims(masks, -1)
        self.inputs = np.concatenate((images, masks,), axis=-1)
        self.inputs = np.transpose(self.inputs, [0,3,1,2]) # for pytorch style inputs (batch x C x H x W)
        self.labels = footprints

    def save_data(self, path):
        index = len(os.listdir(path+"inputs"))
        for i in range(len(self.inputs)):
            np.save(path + "inputs/input_" + str(i+index), self.inputs[i])
            np.save(path + "labels/label_" + str(i+index), self.labels[i])

if __name__=='__main__':
    dataprocessor = DataProcessor()
    dataprocessor.load_raw_data(path="./data")
    dataprocessor.prepare_data(output_shape=(128, 128))
    dataprocessor.save_data("./data/training_data/")
