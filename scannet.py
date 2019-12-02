"""
Mask R-CNN
Train on the SCANNET dataset.

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


class ScannetConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "scannet"

    # Depth Prediction options
    PREDICT_DEPTH = True
    DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    DEPTH_THRESHOLD = 0.1
    USE_ROI_MINI_MASK = False
    PREDICT_NORMAL = False

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    BATCH_SIZE = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 40  # Background + nyu

    TRAIN_ROIS_PER_IMAGE = 200

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BILINEAR_UPSAMPLING = True


############################################################
#  Dataset
############################################################

class ScannetDataset(utils.Dataset):

    def load_scannet(self, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.

        class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair','sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floo mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'showe curtain', 'box', 'whiteboard', 'person', 'nigh stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture',
                       'otherprop']

        n = len(class_names)
        for i in range(n):
            self.add_class("scannet", i+1, class_names[i])

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]


        scannet_data = '/home/orneke/SCANNET/'


        if subset == "train":
            file1 = open(scannet_data + 'scannetv1_train.txt', "r")
            scenes = file1.readlines()
        elif subset == 'val':
            file1 = open(scannet_data + 'scannetv1_val.txt', "r")
            scenes = file1.readlines()
        elif subset == 'test':
            file1 = open(scannet_data + 'scannetv1_test.txt', "r")
            scenes = file1.readlines()

        scannet_data = scannet_data + 'scannet_frames_25k/'
        scenes = [scannet_data + val.strip('\n') + "/" for val in scenes]
        # print(len(scenes))

        #if subset == "train":
        #    scannet_data = scannet_data + 'scannet_frames_25k/'
        #elif subset == 'val' or subset == 'test':
        #    scannet_data = scannet_data + 'scannet_frames_test/'
        #scenes = []
        #for f_name in os.listdir(scannet_data):
        #    if f_name.startswith('scene'):
        #        dir_name = scannet_data + f_name + '/'
        #        scenes.append(dir_name)
        #print(len(scenes))

        i = 0
        for s in scenes:
            for img in os.listdir(s+'color/'):
                try:
                    id = img.split('.')[0]
                    image_path = s + 'color/' + img
                    depth_path = s + 'depth/'+ str(id) + '.png'
                    instance_path = s + 'instance/' + str(id) + '.png'
                    label_path = s + 'label/' + str(id) + '.png'
                    height, width = 968, 1296

                    if os.path.exists(depth_path):
                        self.add_image(
                            "scannet",
                            image_id=i,  # use file number
                            path=image_path,
                            depth_path=depth_path,
                            instance_path=instance_path,
                            label_path=label_path,
                            width=width, height=height,
                            class_ids=None)
                    i += 1
                except:
                    print("Cannot read data ")
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
        if info["source"] != "scannet":
            return super(self.__class__, self).load_mask(image_id)

        label_path = info['label_path']
        instance_path = info['instance_path']

        label = skimage.io.imread(label_path)
        instance = skimage.io.imread(instance_path)

        masks, class_ids = getInstanceMasks(label, instance)

        self.image_info[image_id]['class_ids'] = class_ids

        #? masks = masks.transpose([1, 2, 0])

        return masks.astype(np.bool), class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "scannet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)




def getInstanceMasks(label, instance):
	'''
	Extracts the binary masks for each instance given in a labeled image.

	:param label: labels matrix
	:type label: [H, W], uint16
	:param instance: instance matrix
	:type instance: [H, W], uint16
	:return: binary masks [H, W, N] for each instance N,
			 labels of the instances [N]
	:rtype:
	'''

	H = instance.shape[0]
	W = instance.shape[1]

	pairs = np.unique(np.array([label.flatten(), instance.flatten()]), axis=1)
	pairs = np.transpose(pairs)
	# Remove zero sum rows
	pairs = pairs[~np.all(pairs == 0, axis=1)]

	N = pairs.shape[0]
	instanceMasks = np.zeros([H, W, N])
	instanceLabels = np.zeros([N])

	for i in range(N):
		instanceMasks[:, :, i] = np.logical_and(label == pairs[i, 0], instance == pairs[i, 1])
		instanceLabels[i] = pairs[i, 0]

	return instanceMasks, instanceLabels.astype(np.int32)