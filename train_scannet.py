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


# Train for attention, no shift prediction
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

    config.DEPTH_ATTENTION = True
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = True

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    #model_maskdepth.load_weights(resnet_path)
    #model_maskdepth.load_weights(coco_path, iscoco=False)

    # Call checkpoint below in train_model call!

    #model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    #checkpoint_dir = 'checkpoints/scannet20200115T0147/mask_depth_rcnn_scannet_0002.pth'

    ### Load maskrcnn weights
    checkpoint_dir = 'checkpoints/scannet20200124T1510/mask_depth_rcnn_scannet_0010.pth'
    checkpoint = torch.load(checkpoint_dir)

    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                  'depth' not in k and 'depthmask' not in k and 'depth_ref' not in k and 'depthmask' not in k}

    state = model_maskdepth.state_dict()
    state.update(state_dict)
    model_maskdepth.load_state_dict(state, strict=False)

    ### Load global depth weights
    depth_checkpoint_dir = 'checkpoints/scannet20200202T1605/mask_rcnn_scannet_0015.pth'
    depth_checkpoint = torch.load(depth_checkpoint_dir)

    state_dict = {k: v for k, v in depth_checkpoint['model_state_dict'].items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)

    start = timer()

    epochs = 10
    layers = "depth"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/5, epochs=epochs,
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


# Train for shift heads and roi depths, no attention
def train_roidepth_shift(augmentation=None, depth_weight=1):

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

    config.PREDICT_SHIFT = True
    config.DEPTH_ATTENTION = False


    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    #model_maskdepth.load_weights(resnet_path)
    #model_maskdepth.load_weights(coco_path, iscoco=True)

    # Call checkpoint below in train_model call!

    #model_maskdepth.load_state_dict(torch.load(checkpoint_dir))


    ### Load maskrcnn weights
    checkpoint_dir = 'checkpoints/scannet20200124T1510/mask_depth_rcnn_scannet_0010.pth'
    checkpoint = torch.load(checkpoint_dir)

    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if 'depth' not in k and 'depthmask' not in k and 'depth_ref' not in k and 'depthmask' not in k}

    state = model_maskdepth.state_dict()
    state.update(state_dict)
    model_maskdepth.load_state_dict(state, strict=False)

    ### Load global depth weights
    depth_checkpoint_dir = 'checkpoints/scannet20200202T1605/mask_rcnn_scannet_0015.pth'
    depth_checkpoint = torch.load(depth_checkpoint_dir)

    state_dict = {k: v for k, v in depth_checkpoint['model_state_dict'].items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)


    start = timer()

    epochs = 10
    layers = "depth"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/5, epochs=epochs,
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

    epochs = 20
    layers = "heads"  # options: 3+, 4+, 5+, heads, all

    coco_path = '../mask_rcnn_coco.pth'
    resnet_path = '../resnet50_imagenet.pth'

    depth_model = modellib.DepthCNN(config)
    depth_model.cuda()

    depth_model.load_weights(coco_path)

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
    config.USE_MINI_MASK = True
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

def evaluate_true_roiregions():
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
    config.USE_MINI_MASK = True
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

    checkpoint_dir = 'checkpoints/scannet20200202T1605/mask_rcnn_scannet_0020.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    step = 0
    steps = 5 #5435

    errors = []
    gt_masked_errs = []
    num_pixels = []
    gt_masked_pixels = []

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

            pred_depth = depth_np[0]

            depth_pred = pred_depth.detach().cpu().numpy()[0, 80:560, :]
            depth_gt = gt_depth.cpu().numpy()[0, 80:560, :]

            # Evaluate Global Depth
            err = evaluateDepthsTrue(depth_pred, depth_gt, printInfo=False)
            errors.append(err[:-1])
            num_pixels.append(err[-1])


            # Evaluate GT Masked Regions on Global Depth, ground truth masks
            # Select the nonzero gt masks, depths, rois
            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_masks = gt_masks.data.cpu().numpy()
            gt_boxes = trim_zeros(gt_boxes[0])
            gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
            # Expand the masks from 56x56 to 640x640
            expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

            gt_masked_err, gt_masked_pix = evaluateMaskedRegions(depth_pred, depth_gt, expanded_mask)
            gt_masked_errs += gt_masked_err
            gt_masked_pixels += gt_masked_pix

            step += 1
            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break

    # True errors
    total_pixels = np.sum(num_pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(num_pixels)):
        no_pixels = num_pixels[i]
        rel += errors[i][0] * no_pixels / total_pixels
        relsqr += errors[i][1] * no_pixels / total_pixels
        log10 += errors[i][2] * no_pixels / total_pixels
        rmse += np.power(errors[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errors[i][4], 2) * no_pixels / total_pixels
        a1 += errors[i][5] * no_pixels / total_pixels
        a2 += errors[i][6] * no_pixels / total_pixels
        a3 += errors[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    # Averaged errors
    e = [sum(i) / step for i in zip(*errors)]
    # e = [err/step for err in errors]
    # e = errors / step
    print("Averaged Errors:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    # True errors on GT Masked Regions
    total_pixels = np.sum(gt_masked_pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(gt_masked_pixels)):
        no_pixels = gt_masked_pixels[i]
        rel += gt_masked_errs[i][0] * no_pixels / total_pixels
        relsqr += gt_masked_errs[i][1] * no_pixels / total_pixels
        log10 += gt_masked_errs[i][2] * no_pixels / total_pixels
        rmse += np.power(gt_masked_errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(gt_masked_errs[i][4], 2) * no_pixels / total_pixels
        a1 += gt_masked_errs[i][5] * no_pixels / total_pixels
        a2 += gt_masked_errs[i][6] * no_pixels / total_pixels
        a3 += gt_masked_errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on GT Masked Regions: " , total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))
    # Averaged errors on GT Masked regions
    total_masks = len(gt_masked_pixels)
    e = [sum(i) / total_masks for i in zip(*gt_masked_errs)]

    print("Averaged Errors on GT Masked Regions:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    # Errors on masks of different scales: decide by number of pixels
    # normally area = small < 32^2 (1024) < med < 96^2 (9233) < large
    s, m = 32*32, 96*96

    gt_masked_pixels = np.array(gt_masked_pixels)
    gt_masked_errs = np.array(gt_masked_errs)
    small_ind = (gt_masked_pixels <= s)
    med_ind = np.logical_and((gt_masked_pixels > s), (gt_masked_pixels < m))
    large_ind = (gt_masked_pixels > m)
    small_masks = gt_masked_pixels[small_ind]
    medium_masks = gt_masked_pixels[med_ind]
    large_masks = gt_masked_pixels[large_ind]

    print("small, med, large masks : ", len(small_masks), len(medium_masks), len(large_masks))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(small_masks)
    errs = gt_masked_errs[small_ind]
    for i in range(len(small_masks)):
        no_pixels = small_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Small Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(medium_masks)
    errs = gt_masked_errs[med_ind]
    for i in range(len(medium_masks)):
        no_pixels = medium_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Medium Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(large_masks)
    errs = gt_masked_errs[large_ind]
    for i in range(len(large_masks)):
        no_pixels = large_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Large Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

def evaluate_solodepth():

    print("Model evaluation on scannet test!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test")
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))


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

    checkpoint_dir = 'checkpoints/scannet20200202T1605/mask_rcnn_scannet_0020.pth'
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
        gt_depths = inputs[1]

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

def evaluate_true_attention():
    print("Model evaluation for training set!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("val")  # change the number of steps as well
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

    config.DEPTH_ATTENTION = True
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = True

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20200208T2307/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436
    errors = []
    masked_errs = []
    gt_masked_errs = []
    roi_errs = []
    num_pixels = []
    masked_pixels = []
    gt_masked_pixels = []
    roi_pixels = []

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
                model_maskdepth.predict3([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
                                         mode='inference')

            depth_pred = global_depth.detach().cpu().numpy()[0, 80:560, :]
            depth_gt = gt_depth.cpu().numpy()[0, 80:560, :]

            # Evaluate Global Depth
            err = evaluateDepthsTrue(depth_pred, depth_gt, printInfo=False)
            errors.append(err[:-1])
            num_pixels.append(err[-1])


            # Get normal expanded masks
            detections = detections.data.cpu().numpy()
            window = (0, 0, 480, 640)
            image_shape = config.IMAGE_SHAPE
            mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            mrcnn_depth = mrcnn_depth.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            final_rois, final_class_ids, final_scores, final_masks, final_depths, final_normals = \
                model_maskdepth.unmold_detections(config, detections[0], mrcnn_mask[0],
                                       mrcnn_depth[0], mrcnn_normals[0], image_shape, window)


            # Evaluate Pred Masked Regions on Global Depth, predicted masks
            # final masks = 640x640xROIs
            masked_err, masked_pix = evaluateMaskedRegions(depth_pred, depth_gt, final_masks)
            masked_errs += masked_err
            masked_pixels += masked_pix

            # Evaluate GT Masked Regions on Global Depth, ground truth masks
            # Select the nonzero gt masks, depths, rois
            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_class_ids = gt_class_ids.data.cpu().numpy()
            gt_masks = gt_masks.data.cpu().numpy()
            gt_boxes = trim_zeros(gt_boxes[0])
            gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
            # Expand the masks from 56x56 to 640x640
            expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

            gt_masked_err, gt_masked_pix = evaluateMaskedRegions(depth_pred, depth_gt, expanded_mask)
            gt_masked_errs += gt_masked_err
            gt_masked_pixels += gt_masked_pix

            # Evaluate Mask Accuracy (ROIs)



            #gt_match, pred_match, overlaps = compute_matches(
            #    gt_boxes, gt_class_ids, gt_masks,
            #    pred_boxes, pred_class_ids, pred_scores, pred_masks,
            #    iou_threshold=0.5)

            # Evaluate Prediction Heads (Local Depth)
            # Needs post processing, selecting class ids etc
            # roi_err = evaluateRois(depth_pred, depth_gt)
            #roi_errs.append(roi_err[:-1])
            #roi_pixels.append(roi_err[-1])


            step += 1
            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break

    # True errors
    total_pixels = np.sum(num_pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(num_pixels)):
        no_pixels = num_pixels[i]
        rel += errors[i][0] * no_pixels / total_pixels
        relsqr += errors[i][1] * no_pixels / total_pixels
        log10 += errors[i][2] * no_pixels / total_pixels
        rmse += np.power(errors[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errors[i][4], 2) * no_pixels / total_pixels
        a1 += errors[i][5] * no_pixels / total_pixels
        a2 += errors[i][6] * no_pixels / total_pixels
        a3 += errors[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    # Averaged errors
    e = [sum(i) / step for i in zip(*errors)]
    # e = [err/step for err in errors]
    # e = errors / step
    print("Averaged Errors:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    # True errors on Pred Masked regions
    total_pixels = np.sum(masked_pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(masked_pixels)):
        no_pixels = masked_pixels[i]
        rel += masked_errs[i][0] * no_pixels / total_pixels
        relsqr += masked_errs[i][1] * no_pixels / total_pixels
        log10 += masked_errs[i][2] * no_pixels / total_pixels
        rmse += np.power(masked_errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(masked_errs[i][4], 2) * no_pixels / total_pixels
        a1 += masked_errs[i][5] * no_pixels / total_pixels
        a2 += masked_errs[i][6] * no_pixels / total_pixels
        a3 += masked_errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)


    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Pred Masked Regions: ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    # Averaged errors on Pred Masked regions
    total_masks = len(masked_pixels)
    e = [sum(i) / total_masks for i in zip(*masked_errs)]

    print("Averaged Errors on Pred Masked Regions:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    # True errors on GT Masked Regions
    total_pixels = np.sum(gt_masked_pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(gt_masked_pixels)):
        no_pixels = gt_masked_pixels[i]
        rel += gt_masked_errs[i][0] * no_pixels / total_pixels
        relsqr += gt_masked_errs[i][1] * no_pixels / total_pixels
        log10 += gt_masked_errs[i][2] * no_pixels / total_pixels
        rmse += np.power(gt_masked_errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(gt_masked_errs[i][4], 2) * no_pixels / total_pixels
        a1 += gt_masked_errs[i][5] * no_pixels / total_pixels
        a2 += gt_masked_errs[i][6] * no_pixels / total_pixels
        a3 += gt_masked_errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on GT Masked Regions: " , total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))
    # Averaged errors on GT Masked regions
    total_masks = len(gt_masked_pixels)
    e = [sum(i) / total_masks for i in zip(*gt_masked_errs)]

    print("Averaged Errors on GT Masked Regions:")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    # Errors on masks of different scales: decide by number of pixels
    # normally area = small < 32^2 (1024) < med < 96^2 (9233) < large
    s, m = 32*32, 96*96

    gt_masked_pixels = np.array(gt_masked_pixels)
    gt_masked_errs = np.array(gt_masked_errs)
    small_ind = (gt_masked_pixels <= s)
    med_ind = np.logical_and((gt_masked_pixels > s), (gt_masked_pixels < m))
    large_ind = (gt_masked_pixels > m)
    small_masks = gt_masked_pixels[small_ind]
    medium_masks = gt_masked_pixels[med_ind]
    large_masks = gt_masked_pixels[large_ind]

    print("small, med, large masks : ", len(small_masks), len(medium_masks), len(large_masks))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(small_masks)
    errs = gt_masked_errs[small_ind]
    for i in range(len(small_masks)):
        no_pixels = small_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Small Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(medium_masks)
    errs = gt_masked_errs[med_ind]
    for i in range(len(medium_masks)):
        no_pixels = medium_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Medium Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))

    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    total_pixels = np.sum(large_masks)
    errs = gt_masked_errs[large_ind]
    for i in range(len(large_masks)):
        no_pixels = large_masks[i]
        rel += errs[i][0] * no_pixels / total_pixels
        relsqr += errs[i][1] * no_pixels / total_pixels
        log10 += errs[i][2] * no_pixels / total_pixels
        rmse += np.power(errs[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errs[i][4], 2) * no_pixels / total_pixels
        a1 += errs[i][5] * no_pixels / total_pixels
        a2 += errs[i][6] * no_pixels / total_pixels
        a3 += errs[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("True Errors on Large Masked Regions (GT): ", total_pixels)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))


def evaluate_true_shift():
    print("Model evaluation for training set!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test")  # change the number of steps as well
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

    config.DEPTH_ATTENTION = True
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = True

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20200208T2307/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436
    errors = []
    masked_errs = []
    gt_masked_errs = []
    roi_errs = []
    shift_errs = []
    num_pixels = []
    masked_pixels = []
    gt_masked_pixels = []
    roi_pixels = []
    shift_pixels = []

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

            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, rois, target_depths, mrcnn_depths, target_normal, mrcnn_normal, global_depth, n_dets_per_sample, target_shift, mrcnn_shift = \
                model_maskdepth.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
                                        mode='training')

            depth_pred = global_depth.detach().cpu().numpy()[0, 80:560, :]
            depth_gt = gt_depth.cpu().numpy()[0, 80:560, :]

            ### Evaluate Global Depth
            err = evaluateDepthsTrue(depth_pred, depth_gt, printInfo=False)
            errors.append(err[:-1])
            num_pixels.append(err[-1])

            ### Evaluate Local Depths
            # select nonzero values, class ids
            positive_ix = torch.nonzero(target_class_ids[0] > 0)[:, 0]
            positive_class_ids = target_class_ids[0, positive_ix.data].long()
            indices = torch.stack((positive_ix, positive_class_ids), dim=1)

            ## Gather the depths (predicted and true) that contribute to loss
            target_depths = target_depths.cpu().numpy()
            mrcnn_depths = mrcnn_depths.cpu().numpy()
            target_shift = target_shift.cpu().numpy()
            mrcnn_shift = mrcnn_shift.cpu().numpy()

            true_depth = target_depths[0, indices[:, 0].data, :, :]
            pred_depth = mrcnn_depths[0, indices[:, 0].data, indices[:, 1].data, :, :]
            true_shift = target_shift[0, indices[:, 0].data]
            pred_shift = mrcnn_shift[0, indices[:, 0].data]

            ### Evaluate roi depths
            roi_err = evaluateRoiDepths(pred_depth, true_depth)
            roi_errs.append(roi_err[:-1])
            roi_pixels.append(roi_err[-1])

            ### Evaluate shifts ( amount of shift, absrel, rmse )
            shift_err = evaluateDepthsTrue(pred_shift, true_shift)
            rel, relsqr, rmse, rmse_log = shift_err[0], shift_err[1], shift_err[2], shift_err[4]
            shift_errs.append([rel, relsqr, rmse, rmse_log])
            shift_pixels.append(shift_err[-1]) # number of shifts(ROIs) for one instance


            ### Evaluate local+global by predicted SHIFT, pred boxes
            # We need the unnormalized box coordinates for that, which comes from inference code.

            detections, mrcnn_mask, mrcnn_depth, mrcnn_normal, global_depth, mrcnn_shift = model_maskdepth.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
                                        mode='inference')

            # Replace the locals depths on global by [predicted masks] ?

            # Evaluate whole depth

            # Get masks

            # Evaluate masked regions [pred and gt masks]

            ### Evaluate local+global by GT SHIFT, pred boxes
            # a. Whole depth
            # b. Masked regions


            # Get normal expanded masks
            detections = detections.data.cpu().numpy()
            window = (0, 0, 480, 640)
            image_shape = config.IMAGE_SHAPE
            mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            mrcnn_depth = mrcnn_depth.permute(0, 1, 3, 4, 2).data.cpu().numpy()
            final_rois, final_class_ids, final_scores, final_masks, final_depths, final_normals = \
                model_maskdepth.unmold_detections(config, detections[0], mrcnn_mask[0],
                                                  mrcnn_depth[0], mrcnn_normal[0], image_shape, window)

            # Evaluate Pred Masked Regions on Global Depth, predicted masks
            # final masks = 640x640xROIs
            masked_err, masked_pix = evaluateMaskedRegions(depth_pred, depth_gt, final_masks)
            masked_errs += masked_err
            masked_pixels += masked_pix

            # Evaluate GT Masked Regions on Global Depth, ground truth masks
            # Select the nonzero gt masks, depths, rois
            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_class_ids = gt_class_ids.data.cpu().numpy()
            gt_masks = gt_masks.data.cpu().numpy()
            gt_boxes = trim_zeros(gt_boxes[0])
            gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
            # Expand the masks from 56x56 to 640x640
            expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

            gt_masked_err, gt_masked_pix = evaluateMaskedRegions(depth_pred, depth_gt, expanded_mask)
            gt_masked_errs += gt_masked_err
            gt_masked_pixels += gt_masked_pix



            step += 1
            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break


def count_objectregions():

    print("ROI training!")

    config = scannet.ScannetConfig()

    dataset_train = scannet.ScannetDataset()
    dataset_train.load_scannet("train")
    dataset_train.prepare()

    dataset_val = scannet.ScannetDataset()
    dataset_val.load_scannet("val")
    dataset_val.prepare()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test")  # change the number of steps as well
    dataset_test.prepare()

    print("--TRAIN--")
    print("Image Count: {}".format(len(dataset_train.image_ids)))
    print("Class Count: {}".format(dataset_train.num_classes))

    print("--VAL--")
    print("Image Count: {}".format(len(dataset_val.image_ids)))
    print("Class Count: {}".format(dataset_val.num_classes))

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))

    config.STEPS_PER_EPOCH = 1
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 0

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.PREDICT_NORMAL = False
    config.PREDICT_PLANE = False
    # config.USE_MINI_MASK = True
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.GRAD_LOSS = False
    config.BATCH_SIZE = 1

    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28

    config.IMAGE_MAX_DIM = 640

    train_generator = data_generator2(dataset_train, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)
    val_generator = data_generator2(dataset_val, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    train_steps, val_steps, test_steps = 16830, 2635, 5436

    train_pixels, val_pixels, test_pixels = [], [], []

    for i in range(train_steps):

        if i % 100 == 0:
            print(i)

        inputs = next(train_generator)

        images = inputs[0]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        from torch.autograd import Variable

        # Wrap in variables
        images = Variable(images)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        images = images.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_masks = gt_masks.cuda()

        gt_boxes = gt_boxes.data.cpu().numpy()
        gt_masks = gt_masks.data.cpu().numpy()
        gt_boxes = trim_zeros(gt_boxes[0])
        N = gt_boxes.shape[0]
        for i in range(N):
            no_pix = float(gt_masks[0, i, :, :].sum())
            train_pixels.append(no_pix)


    for i in range(val_steps):

        if i % 100 == 0:
            print(i)

        inputs = next(val_generator)

        images = inputs[0]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        from torch.autograd import Variable

        # Wrap in variables
        images = Variable(images)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        images = images.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_masks = gt_masks.cuda()

        gt_boxes = gt_boxes.data.cpu().numpy()
        gt_masks = gt_masks.data.cpu().numpy()
        gt_boxes = trim_zeros(gt_boxes[0])
        N = gt_boxes.shape[0]
        for i in range(N):
            no_pix = float(gt_masks[0, i, :, :].sum())
            val_pixels.append(no_pix)

    for i in range(test_steps):

        if i % 100 == 0:
            print(i)

        inputs = next(test_generator)

        images = inputs[0]
        gt_boxes = inputs[5]
        gt_masks = inputs[6]

        from torch.autograd import Variable

        # Wrap in variables
        images = Variable(images)
        gt_boxes = Variable(gt_boxes)
        gt_masks = Variable(gt_masks)

        images = images.cuda()
        gt_boxes = gt_boxes.cuda()
        gt_masks = gt_masks.cuda()

        gt_boxes = gt_boxes.data.cpu().numpy()
        gt_masks = gt_masks.data.cpu().numpy()
        gt_boxes = trim_zeros(gt_boxes[0])
        N = gt_boxes.shape[0]
        for i in range(N):
            no_pix = float(gt_masks[0, i, :, :].sum())
            test_pixels.append(no_pix)

    print("Total num of masks, train, val, test: ", len(train_pixels), len(val_pixels), len(test_pixels))

    s, m = 32 * 32, 96 * 96

    train_pixels = np.array(train_pixels)
    small_ind_train = (train_pixels <= s)
    med_ind_train = np.logical_and((train_pixels > s), (train_pixels < m))
    large_ind_train = (train_pixels > m)

    val_pixels = np.array(val_pixels)
    small_ind_val = (val_pixels <= s)
    med_ind_val = np.logical_and((val_pixels > s), (val_pixels < m))
    large_ind_val = (val_pixels > m)

    test_pixels = np.array(test_pixels)
    small_ind_test = (test_pixels <= s)
    med_ind_test = np.logical_and((test_pixels > s), (test_pixels < m))
    large_ind_test = (test_pixels > m)

    print("Train set, small, med, large layers:", np.sum(small_ind_train), np.sum(med_ind_train), np.sum(large_ind_train))
    print("Val set, small, med, large layers:", np.sum(small_ind_val), np.sum(med_ind_val), np.sum(large_ind_val))
    print("Test set, small, med, large layers:", np.sum(small_ind_test), np.sum(med_ind_test), np.sum(large_ind_test))

    tot_small = np.sum(small_ind_train)+np.sum(small_ind_val)+np.sum(small_ind_test)
    tot_med = np.sum(med_ind_train) + np.sum(med_ind_val) + np.sum(med_ind_test)
    tot_large = np.sum(large_ind_train) + np.sum(large_ind_val) + np.sum(large_ind_test)

    print("Total set, small, med, large layers:", tot_small, tot_med, tot_large)


    print("Train total num of pixels small, med, large layers:", np.sum(train_pixels[small_ind_train]), np.sum(train_pixels[med_ind_train]), np.sum(train_pixels[large_ind_train]))

    print("Val total num of pixels small, med, large layers:", np.sum(val_pixels[small_ind_val]),
          np.sum(val_pixels[med_ind_val]), np.sum(val_pixels[large_ind_val]))

    print("Test total num of pixels small, med, large layers:", np.sum(test_pixels[small_ind_test]),
          np.sum(test_pixels[med_ind_test]), np.sum(test_pixels[large_ind_test]))


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

        #train_roidepth_shift(augmentation)

        #evaluate_roiregions()

        #train_roidepth(augmentation)

        #evaluate_true_attention()

        #evaluate_true_roiregions()
        #evaluate_true_shift()

        count_objectregions()
        #train_roidepth(augmentation)
        #train_solodepth(augmentation)

        #evaluate_roidepth()
        #evaluate_roidepth()



