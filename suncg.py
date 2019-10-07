"""
Mask R-CNN
Train on the SUN dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Mask RCNN is Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sun.py train --dataset=/Users/ekaterina/Desktop/diploma/mask_rcnn/datasets/SUNRGBD/train --weights=coco


    # Resume training a model that you had trained earlier
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=last

    # Train a new model starting from ImageNet weights
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=imagenet

    # Apply color splash to an image
    python3 sun.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 sun.py splash --weights=last --video=<URL or path to file>ß
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from config import Config
import utils


############################################################
#  Configurations
############################################################


class SuncgConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "suncg"

    # Depth Prediction options
    PREDICT_DEPTH = True
    DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    DEPTH_THRESHOLD = 0.1
    USE_ROI_MINI_MASK = False
    PREDICT_NORMAL = False

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 37  # Background + suncg

    TRAIN_ROIS_PER_IMAGE = 200

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BILINEAR_UPSAMPLING = True


############################################################
#  Dataset
############################################################

class SuncgDataset(utils.Dataset):

    def load_sun(self, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.

        class_names = ["toilet", "whiteboard", "wall", "otherstructure", "night_stand", "otherprop", "books", "mirror", "table", "chair", "otherfurniture", "floor", "window", "lamp", "refridgerator", "curtain", "blinds", "dresser", "picture", "ceiling", "door", "shower_curtain", "void", "cabinet", "sink", "desk", "bookshelf", "television", "floor_mat", "shelves", "sofa", "counter", "bed", "person", "bathtub", "pillow", "clothes"]

        n = len(class_names)
        for i in range(n):
            self.add_class("suncg", i+1, class_names[i])

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]


        suncg_data = '/home/dhamo/Documents/GLrenderSUNCG/generated_images'
        mask_file = '/home/orneke/SUNCG_masks/train/'

        if subset == "train":
            file1 = open(suncg_data + '/train_split.txt', "r")
            split = file1.readlines()
            mask_file += 'train/'
        elif subset == 'val':
            file1 = open(suncg_data + '/val_split.txt', "r")
            split = file1.readlines()
            mask_file += 'val/'

        split = [int(val[:-1]) for val in split]

        for i in split:
            try:
                file1 = open(suncg_data + '/category_nyu/' + str(i) + '.txt', "r")
                class_ids_str = file1.readlines()[0].split(" ")[:-1]
                class_ids = [int(x)+1 for x in class_ids_str]
                class_ids.append(0) # For background mask

                image_path = suncg_data + '/rgb/' + str(i) + '.png'
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]

                depth_path = suncg_data + '/depth/' + str(i) + '.png'
                #depth_path = os.path.join(dataset_dir, depth_file_name)

                mask_path = mask_file + str(i) + '.npy'

                if os.path.exists(mask_path) and os.path.exists(depth_path):
                    self.add_image(
                        "suncg",
                        image_id=i,  # use file number
                        path=image_path,
                        depth_path=depth_path,
                        mask_path=mask_path,
                        width=width, height=height,
                        class_ids=class_ids)

            except:
                #print("Cannot read data ")
                continue


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "suncg":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = info['class_ids']

        mask_path = info["mask_path"]
        masks = np.load(mask_path)
        masks = masks.transpose([1, 2, 0])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks.astype(np.bool), class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "suncg":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)



