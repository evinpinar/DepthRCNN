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
import scannet
import nyu
from sun import SunConfig
from models.model_maskdepthrcnn import *
from timeit import default_timer as timer
from config import Config
import utils
from tensorboardX import SummaryWriter
import imgaug.augmenters as iaa

from evaluate_utils import *
import logging


def train_roidepth(augmentation=None, depth_weight=1):

    print("ROI training!")

    config = scannet.ScannetConfig()

    dataset_train = scannet.ScannetDataset()
    dataset_train.load_scannet("train")
    dataset_train.prepare()

    dataset_val = scannet.ScannetDataset()
    dataset_val.load_scannet("val")
    dataset_val.prepare()

    print("--TRAIN--")
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))

    print("--VAL--")
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))

    config.STEPS_PER_EPOCH = 15000
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 2000

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    depth_weight = 1
    config.USE_MINI_MASK = True
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    #model_maskdepth.load_weights(resnet_path)
    model_maskdepth.load_weights(coco_path, iscoco=True)


    # Call checkpoint below in train_model call!

    #model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    #checkpoint_dir = 'checkpoints/scannet20200115T0147/mask_depth_rcnn_scannet_0002.pth'

    start = timer()

    epochs = 5
    layers = "maskrcnn"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)

    #model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/2, epochs=epochs,
    #                             layers=layers, depth_weight=depth_weight, augmentation=augmentation,
    #                             checkpoint_dir_prev=checkpoint_dir, continue_train=False)

    '''
    epochs = 40
    layers = "all"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight,
                                 augmentation=augmentation, continue_train=True)

    epochs = 60
    layers = "5+"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight,
                                 augmentation=augmentation, continue_train=True)

    '''
    '''

    epochs = 40
    layers = "all"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)


    epochs = 60
    config.LEARNING_RATE /= 10
    layers = "5+"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation)

    '''

    end = timer()
    print('Total training time: ', (end - start)/60)


def train_solodepth(augmentation=None):

    print("FPN Depth Baseline0 training!")

    config = scannet.ScannetConfig()

    dataset_train = scannet.ScannetDataset()
    dataset_train.load_scannet("train")
    dataset_train.prepare()

    dataset_val = scannet.ScannetDataset()
    dataset_val.load_scannet("val")
    dataset_val.prepare()

    print("--TRAIN--")
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))

    print("--VAL--")
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))

    config.BATCH_SIZE = 6
    config.STEPS_PER_EPOCH = 2500 # 15000 changed for batch size 6
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 330 # 2000

    config.PREDICT_DEPTH = True
    config.GRAD_LOSS = False
    config.DEPTH_THRESHOLD = 0
    config.DEPTH_LOSS = 'L1'

    epochs = 10
    layers = "heads"  # options: 3+, 4+, 5+, heads, all

    coco_path = '../mask_rcnn_coco.pth'
    resnet_path = '../resnet50_imagenet.pth'
    resnet_path2 = '../e2e_mask_rcnn_R_50_FPN_1x.pth'

    depth_model = modellib.DepthCNN(config)
    depth_model.cuda()

    depth_model.load_weights(resnet_path2)

    start = timer()
    depth_model.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, augmentation=augmentation)

    end = timer()
    print('Total training time: ', end - start)

def evaluate_roidepth():

    print("Model evaluation for training set!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("val") # change the number of steps as well
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))


    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    depth_weight = 1
    config.USE_MINI_MASK = True
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20191218T1651/mask_depth_rcnn_scannet_0014.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    #test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1, augmentation=None)
    errors = np.zeros(8)
    step = 0
    steps = 2635
    # train = 16830, val = 2635, test = 5436

    with torch.no_grad():
        while step < steps:
            inputs = next(test_generator)
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_depths = inputs[7]
            gt_normals = inputs[8]
            gt_depth = inputs[9]

            # Wrap in variables
            images = Variable(images)
            rpn_match = Variable(rpn_match)
            rpn_bbox = Variable(rpn_bbox)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            gt_depths = Variable(gt_depths)
            gt_normals = Variable(gt_normals)
            gt_depth = Variable(gt_depth)

            # To GPU
            if config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_depths = gt_depths.cuda()
                gt_normals = gt_normals.cuda()
                gt_depth = gt_depth.cuda()

            detections, mrcnn_mask, mrcnn_depth, mrcnn_normals, global_depth = \
                model_maskdepth.predict3([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals], mode='inference')

            depth_pred = global_depth.detach().cpu().numpy()[0, 80:560, :]
            depth_gt = gt_depth.cpu().numpy()[0, 80:560, :]

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

# Evaluates the masked depth predictions
def evaluate_roiregions():
    print("Model evaluation!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test")
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))

    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    depth_weight = 1
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    # config.USE_MINI_MASK = True
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28


    ''' 
    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20191218T1651/mask_depth_rcnn_scannet_0020.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])
    '''

    depth_model = DepthCNN(config)
    depth_model.cuda()

    checkpoint_dir = 'checkpoints/scannet20191209T1450/mask_rcnn_scannet_0020.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    step = 0
    steps = 5435

    loss_roi = []
    with torch.no_grad():
        while step < steps:
            inputs = next(test_generator)
            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_depths = inputs[7]
            gt_normals = inputs[8]
            gt_depth = inputs[9]

            # Wrap in variables
            images = Variable(images)
            rpn_match = Variable(rpn_match)
            rpn_bbox = Variable(rpn_bbox)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            gt_depths = Variable(gt_depths)
            gt_normals = Variable(gt_normals)
            gt_depth = Variable(gt_depth)

            # To GPU
            if config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_depths = gt_depths.cuda()
                gt_normals = gt_normals.cuda()
                gt_depth = gt_depth.cuda()

            #detections, mrcnn_mask, mrcnn_depth, mrcnn_normals, global_depth = \
            #    model_maskdepth.predict3([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
            #                             mode='inference')

            #images = images.permute(0, 2, 3, 1)
            #r = model_maskdepth.detect(images, mold_image=False, image_metas=image_metas)
            #pred_depth = r[0]["glob_depth"][0]

            depth_np = depth_model.predict([images, gt_depths], mode='inference')


            pred_depth = depth_np[0][0]
            rois = gt_boxes[0]
            masks = gt_masks[0]
            gt_dep = gt_depth[0]




            roi_err = eval_roi_accuracy(gt_dep, pred_depth, rois, masks)
            #print(roi_err)
            loss_roi.append(np.mean(roi_err))
            step += 1

            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break

    print("Masked regions depth mean: ", np.mean(loss_roi))


def evaluate_solodepth():

    print("Model evaluation on scannet test!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("val")
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))

    test_generator = data_generator_onlydepth(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                              augmentation=None)


    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.DEPTH_THRESHOLD = 0
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    config.USE_MINI_MASK = False
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28

    depth_model = DepthCNN(config)
    depth_model.cuda()

    checkpoint_dir = 'checkpoints/scannet20191209T1450/mask_rcnn_scannet_0020.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    test_generator = data_generator_onlydepth(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                              augmentation=None)

    errors = []
    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436

    while step < steps:

        inputs = next(test_generator)
        images = inputs[0]
        gt_depths = inputs[2]

        images = Variable(images)
        gt_depths = Variable(gt_depths)

        images = images.cuda()
        gt_depths = gt_depths.cuda()

        depth_np = depth_model.predict([images, gt_depths], mode='inference')

        depth_pred = depth_np[0][0, 80:560, :].detach().cpu().numpy()
        depth_gt = gt_depths[0, 80:560, :].cpu().numpy()

        err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
        errors.append(err)
        step += 1

        if step % 100 == 0:
            print(" HERE: ", step)
            print(err)

    e = np.array(errors).mean(0).tolist()
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
        # iaa.Sometimes(0.5,iaa.GaussianBlur(sigma=(0, 0.25))),
        # Strengthen or weaken the contrast in each image.
        # iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        # iaa.Multiply((0.8, 1.2)),
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

        #evaluate_roidepth()
        #evaluate_solodepth()

        #evaluate_roiregions()

        #train_roidepth(augmentation)

        train_roidepth(augmentation)
        #train_solodepth(augmentation)

        #evaluate_roidepth()
        #evaluate_roidepth()



