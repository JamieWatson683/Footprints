import numpy as np
import cv2
from skimage import io, color, measure, morphology
import os


def extract_footprint(image, template, mask, return_overlay=False):
    """"""

    # Normalise image if necessary
    if image.max() > 1:
        image = image / 255

    # Convert to appropriate type
    image = image.astype(np.float32)
    image[150:,:] = 0
    template = template.astype(np.float32)

    # Find template in image
    result = cv2.matchTemplate(image, template, 0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Add mask to array
    footprint = np.zeros((image.shape[0], image.shape[1]))
    footprint[min_loc[1]:min_loc[1] + mask.shape[0], min_loc[0]:min_loc[0] + mask.shape[1]] = mask

    # Optionally return overlay with footprint array
    if return_overlay:
        overlay = color.label2rgb(footprint, image, bg_label=0)
        return footprint, min_loc, overlay
    else:
        return footprint, min_loc


def extract_all_footprints(path, save_footprint=True, save_overlay=True, two_masks=False, mask_cutoff_column=256):
    """"""

    # Load frames, template and mask
    sky_names = os.listdir(path + "sky_frames")
    sky_frames = []
    for name in sky_names:
        sky_frames.append(path + "sky_frames/" + name)
    sky_frames = io.imread_collection(sky_frames).concatenate()

    if two_masks:
        mask1 = io.imread(path + 'mask1.png')[:, :, 0] == 255
        mask2 = io.imread(path + 'mask2.png')[:, :, 0] == 255
        template1 = (io.imread(path + 'template1.png') / 255).astype(np.float32)
        template2 = (io.imread(path + 'template2.png') / 255).astype(np.float32)
    else:
        mask = io.imread(path + 'mask.png')[:, :, 0] == 255
        template = (io.imread(path + 'template.png') / 255).astype(np.float32)


    # Loop through frames - get footprint and overlay
    for index in range(len(sky_frames)):
        name = sky_names[index]
        image = (sky_frames[index] / 255).astype(np.float32)
        #print(name)
        if not two_masks:
            footprint, min_loc, overlay = extract_footprint(image, template, mask, return_overlay=True)
        else:
            footprint1, min_loc1, overlay1 = extract_footprint(image, template1, mask1, return_overlay=True)
            if min_loc1[0] <= mask_cutoff_column:
                footprint2, min_loc2, overlay2 = extract_footprint(image, template2, mask2, return_overlay=True)
            if min_loc1[0] > mask_cutoff_column:
                footprint, overlay = footprint1, overlay1
            elif min_loc1[0] - mask_cutoff_column > -20:
                print("Binary erosion taking place {}".format(name))
                footprint_diff = morphology.binary_erosion((footprint1 - footprint2) > 0)
                footprint, overlay = footprint_diff + footprint2, overlay2
            # elif min_loc1[0] - mask_cutoff_column > -40:
            #     print("Binary erosion taking place {}".format(name))
            #     footprint_diff = morphology.binary_erosion((footprint1 - footprint2) > 0)
            #     footprint_diff = morphology.binary_erosion(footprint_diff)
            #     footprint, overlay = footprint_diff + footprint2, overlay2
            else:
                footprint, overlay = footprint2, overlay2

            # Save footprint and overlay
        if save_footprint:
            np.save(path+"sky_footprints/footprints/"+name[:-4], footprint)

        if save_overlay:
            io.imsave(path + "sky_footprints/overlays/" + name, overlay)


def place_footprint(image, sky_footprint, homography):

    # Find points and create homogenous coordinates
    mask_points = np.nonzero(sky_footprint)
    mask_points = np.transpose(mask_points)
    homogenous_points = np.ones((mask_points.shape[0], 3))
    homogenous_points[:, 0:2] = mask_points

    # Compute mapping
    mapped_points = np.matmul(homography, homogenous_points.T)
    mapped_points = mapped_points / mapped_points[2]
    mapped_points = np.round(mapped_points[:2]).astype(int)

    # Create footprint array
    footprint = np.zeros_like(sky_footprint)
    footprint[mapped_points[0, :], mapped_points[1, :]] = 1
    footprint = footprint.astype(int)

    # Fill gaps using convex hull
    label_properties = measure.regionprops(footprint)[0]
    bbox = label_properties.bbox
    hull = label_properties.convex_image
    footprint[bbox[0]:bbox[2], bbox[1]:bbox[3]] = hull

    # Get overlay
    overlay = color.label2rgb(footprint, image, bg_label=0)

    return footprint, overlay


def place_all_footprints(path, homography, save_footprint=True, save_overlay=True):

    # Load names
    ground_names = os.listdir(path + "ground_frames")
    for name in ground_names:
        # Load image
        image = io.imread(path+"ground_frames/"+name)
        # Load footprint
        number = name[12:-4]
        sky_footprint = np.load(path+"sky_footprints/footprints/sky_video"+name[12:-4]+".npy").astype(int)

        ground_footprint, overlay = place_footprint(image, sky_footprint, homography)

        if save_footprint:
            np.save(path+"ground_footprints/footprints/"+name[:-4], ground_footprint)

        if save_overlay:
            io.imsave(path+"ground_footprints/overlays/"+name, overlay)


def save_ground_mask_overlays(path):
    mask_list = os.listdir(path+"ground_masks/masks/")

    for name in mask_list:
        mask = np.load(path+"ground_masks/masks/"+name)
        image = io.imread(path+"ground_frames/"+name[:-4]+".jpg")
        overlay = color.label2rgb(mask, image, bg_label=0)
        io.imsave(path+"ground_masks/overlays/"+name[:-4]+".jpg", overlay)


