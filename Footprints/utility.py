import numpy as np
from skimage import io, transform, measure
import os


class DataLoader(object):
    def __init__(self):
        self.masks = None
        self.footprints = None
        self.images = None
        self.inputs = None
        self.labels = None
        self.filenames = None
        self.bboxes = None  # To store bbox of cropped image for reconstruction

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


    def crop_example(self, mask, image, footprint, output_shape, height_expand=0.3, width_expand=0.3):
        """WARNING DO NOT USE - need to amend to not use footprint for bbox calculation...maybe another network to
        predict this from obscured image"""

        # Get bounding box of mask (for height) and footprint (for width)
        bbox_mask = measure.regionprops(measure.label(mask))[0].bbox
        height_expand = int((bbox_mask[2] - bbox_mask[0]) * height_expand / 2)

        bbox_footprint = measure.regionprops(measure.label(footprint))[0].bbox
        width_expand = int((bbox_footprint[3] - bbox_footprint[1]) * width_expand / 2)

        # Compute expanded bounding box coordinates
        left = max(bbox_footprint[1] - width_expand, 0)
        right = min(bbox_footprint[3] + width_expand, mask.shape[1] - 1)
        top = max(bbox_mask[0] - height_expand, 0)
        bottom = min(bbox_mask[2] + height_expand, mask.shape[0])

        # Crop inputs and resize
        mask = transform.resize(mask[top:bottom, left:right], output_shape=output_shape)
        image = transform.resize(image[top:bottom, left:right, :], output_shape=output_shape)
        footprint = transform.resize(footprint[top:bottom, left:right], output_shape=output_shape)

        # Normalise and threshold
        mask = (mask / mask.max()) > 0.5
        footprint = (footprint / footprint.max()) > 0.5
        image = image / image.max()

        # save new bbox coordinates
        bbox = np.array([top, left, bottom, right])

        return mask, image, footprint, bbox

    def prepare_data(self, output_shape, height_expand=0.3, width_expand=0.3, crop_all=False):

        if crop_all:
            size = len(self.masks)

        # intialise for storage
            cropped_masks = np.zeros([size, output_shape[0], output_shape[1]], dtype=int)
            cropped_footprints = np.zeros_like(cropped_masks)
            cropped_images = np.zeros([size, output_shape[0], output_shape[1], 3])
            bboxes = np.zeros([size, 4], dtype=int)

            for index in range(size):
                # Loop and store cropped data
                cropped_masks[index], cropped_images[index], cropped_footprints[index], bboxes[index] = \
                self.crop_example(self.masks[index], self.images[index], self.footprints[index],
                                  output_shape=output_shape, height_expand=height_expand, width_expand=width_expand)

            # Save in form for network
            cropped_masks = np.expand_dims(cropped_masks, axis=-1)
            self.inputs = np.concatenate((cropped_masks, cropped_images), axis=-1)
            self.labels = cropped_footprints
            self.bboxes = bboxes

        else:
            masks = np.expand_dims(self.masks, -1)
            self.inputs = np.concatenate((self.images, masks,), axis=-1)
            self.inputs = np.transpose(self.inputs, [0,3,1,2]) # for pytorch style inputs (batch x C x H x W)
            self.labels = self.footprints

    def save_data(self, path, name):
        np.save(path+name+"_inputs", self.inputs)
        np.save(path+name+"_labels", self.labels)

if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.load_raw_data('./data/')
    dataloader.prepare_data(output_shape=None)
    dataloader.save_data(path='./data/', name='testdata')
