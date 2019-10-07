import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import zipfile
import urllib.request
import shutil

import torch
import scipy.io
import scipy.misc

import numpy as np
import h5py
from tqdm import tqdm

import models.model as modellib
from models.model import log
import suncg
from sun import SunConfig
from models.model_maskdepthrcnn import *
from timeit import default_timer as timer
from config import Config
import utils
from tensorboardX import SummaryWriter
import imgaug.augmenters as iaa

from evaluate_utils import *


def train_roidepth(augmentation=None, depth_weight=1):
    config = suncg.SuncgConfig()

    dataset_train = suncg.SuncgDataset()
    dataset_train.load_sun("train")
    dataset_train.prepare()

    dataset_val = suncg.SuncgDataset()
    dataset_val.load_sun("val")
    dataset_val.prepare()

    config.STEPS_PER_EPOCH = 9000
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 1000

    config.PREDICT_DEPTH = True
    depth_weight = 1
    config.USE_MINI_MASK = True
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101
    #model_maskdepth.load_weights(resnet_path)
    model_maskdepth.load_weights(coco_path, iscoco=False)

    #checkpoint_dir = 'checkpoints/suncg20190924T2014/mask_depth_rcnn_suncg_0020.pth'
    #model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    start = timer()

    epochs = 20
    layers = "heads"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)


    epochs = 40
    layers = "all"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)


    epochs = 60
    config.LEARNING_RATE /= 10
    layers = "5+"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)


    end = timer()
    print('Total training time: ', end - start)


### TODO: Change this code for suncg
def evaluate_maskrcnn():
    config = sun.SunConfig()
    SUN_DIR_test = '../SUNRGBD/train'

    dataset_test = sun.SunDataset()
    ### Test set size is too big, 5000. Hence I changed to val!
    dataset_test.load_sun(SUN_DIR_test, "val")
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))
    for i, info in enumerate(dataset_test.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    config.PREDICT_DEPTH = True
    mask_model = modellib.MaskRCNN(config)
    mask_model.cuda()

    checkpoint_dir = 'checkpoints/sun20190801T2112/mask_rcnn_sun_0030.pth'
    mask_model.load_state_dict(torch.load(checkpoint_dir))

    test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    errors = np.zeros(8)
    step = 0
    steps = 1000
    while step < steps:
        inputs = next(test_datagenerator)
        images = inputs[0]
        image_metas = inputs[1]
        rpn_match = inputs[2]
        rpn_bbox = inputs[3]
        gt_class_ids = inputs[4]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]
        gt_depths = inputs[7]

        # Wrap in variables
        images = Variable(images)
        rpn_match = Variable(rpn_match)
        rpn_bbox = Variable(rpn_bbox)
        gt_class_ids = Variable(gt_class_ids)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        images = images.cuda()
        rpn_match = rpn_match.cuda()
        rpn_bbox = rpn_bbox.cuda()
        gt_class_ids = gt_class_ids.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_masks = gt_masks.cuda()

        detections, mrcnn_mask, depth_np = mask_model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks],
                                                              mode='inference')

        depth_pred = depth_np.detach().cpu().numpy()[0, 80:560, :]
        depth_gt = gt_depths.cpu().numpy()[0, 80:560, :]

        err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
        errors = errors + err
        step += 1

        if step % 100 == 0:
            print(" HERE: ", step)

        # Break after 'steps' steps
        if step == steps - 1:
            break

    e = errors / step
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))


if __name__ == '__main__':
    augmentation = iaa.Sometimes(.667, iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.25.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.25))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2)),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        # iaa.Affine(
        #	scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #	translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        #	rotate=(-5, 5),
        #	#shear=(-8, 8)
        # iaa.Crop(percent=(0, 0.1)),  # random crops
        # )
    ], random_order=True))  # apply augmenters in random order

    import warnings

    # warnings.filterwarnings("ignore")
    print("starting!")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("ROI training!")
        train_roidepth(augmentation)

