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

import nyu
import models.model as model
from models.model_maskdepthrcnn import *
from timeit import default_timer as timer
import utils
from tensorboardX import SummaryWriter
import imgaug.augmenters as iaa

from evaluate_utils import *
from evaluate_depth import *


def train_maskrcnn(augmentation=None, depth_weight=0):
    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    standard_split = False

    if standard_split:
        dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
        dataset_val = nyu.NYUDataset(path_to_dataset, 'test', config)
    else:
        ### CHANGE NUMPY FILES IN NYU_DATASET ACCORDINGLY!!!
        dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
        dataset_val = nyu.NYUDataset(path_to_dataset, 'val', config)
        dataset_test = nyu.NYUDataset(path_to_dataset, 'test', config)

    config.STEPS_PER_EPOCH = 600
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 100

    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.PREDICT_DEPTH = True
    mask_model = model.MaskRCNN(config)

    resnet_path = '../resnet50_imagenet.pth'
    mask_model.load_weights(resnet_path)

    #checkpoint_dir = 'checkpoints/nyudepth20190722T1403/mask_rcnn_nyudepth_0050.pth'
    #mask_model.load_state_dict(torch.load(checkpoint_dir))

    mask_model.cuda()
    mask_model.train()
    start = timer()

    layers = "heads"  # options: 3+, 4+, 5+, heads, all
    epochs = 50
    mask_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                           layers=layers, depth_weight=depth_weight)


    layers = "all"  # options: 3+, 4+, 5+, heads, all
    epochs = 40
    mask_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                           layers=layers, depth_weight=depth_weight)


    layers = "4+"  # options: 3+, 4+, 5+, heads, all
    epochs = 30
    config.LEARNING_RATE /=10
    mask_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                           layers=layers, depth_weight=depth_weight)

    end = timer()



    print('Total training time: ', end - start)


def train_depth(augmentation=None):

    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    dataset_train = nyu.NYUDepthDataset(path_to_dataset, 'train', config, augment=False, augmentation=augmentation)
    dataset_val = nyu.NYUDepthDataset(path_to_dataset, 'test', config)
    # dataset_test = nyu.NYUDepthDataset(path_to_dataset, 'test', config)


    config.STEPS_PER_EPOCH = 700
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 100
    config.BATCH_SIZE = 1

    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    config.DEPTH_LOSS = 'CHAMFER' # Options: L1, L2, BERHU
    config.CHAM_LOSS = False
    config.CHAM_WEIGHT = 100
    config.GRAD_LOSS = False

    depth_model = model.DepthCNN(config)

    depth_model.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    #depth_model.load_weights(resnet_path)

    checkpoint_dir = 'checkpoints/nyudepth20191207T1213/mask_rcnn_nyudepth_0200.pth'
    #depth_model.load_state_dict(torch.load(checkpoint_dir))

    depth_model.train()
    start = timer()

    epochs = 50
    depth_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)

    #config.LEARNING_RATE /= 10
    #epochs = 200
    #depth_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)


    end = timer()
    print('Total training time: ', end - start)


def train_depth_cham_masked(augmentation=None):

    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
    dataset_val = nyu.NYUDataset(path_to_dataset, 'test', config)
    # dataset_test = nyu.NYUDepthDataset(path_to_dataset, 'test', config)


    config.STEPS_PER_EPOCH = 700
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 100

    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    config.DEPTH_LOSS = 'L1' # Options: L1, L2, BERHU
    config.CHAM_LOSS = False
    config.GRAD_LOSS = False
    config.CHAM_COMBINE = False

    depth_model = model.DepthCNN(config)

    depth_model.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    depth_model.load_weights(resnet_path)

    #checkpoint_dir = 'checkpoints/nyudepth20191003T1927/mask_rcnn_nyudepth_0100.pth'
    #depth_model.load_state_dict(torch.load(checkpoint_dir))

    checkpoint_dir = 'checkpoints/nyudepth20191207T1213/mask_rcnn_nyudepth_0200.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    depth_model.train()
    start = timer()

    epochs = 50
    depth_model.train_model4(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)

    #config.LEARNING_RATE /= 10
    #epochs = 200
    #depth_model.train_model4(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)


    end = timer()
    print('Total training time: ', end - start)


def train_depthrcnn(augmentation=None, depth_weight=0):
    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
    dataset_test = nyu.NYUDataset(path_to_dataset, 'test', config, augmentation=augmentation)

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.GEOMETRIC_LOSS = True
    config.GRAD_LOSS = True
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
    #model_maskdepth.load_weights(resnet_path)

    checkpoint_dir = 'checkpoints/scannet20191218T1651/mask_depth_rcnn_scannet_0020.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    config.STEPS_PER_EPOCH = 700
    config.TRAIN_ROIS_PER_IMAGE = 100
    config.VALIDATION_STEPS = 100

    epochs = 100
    layers = "heads"  # options: 3+, 4+, 5+, heads, all

    start = timer()
    model_maskdepth.train_model2(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                layers=layers)
    end = timer()
    print('Total training time: ', end - start)

def evaluate_solodepth():

    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    dataset_val = nyu.NYUDepthDataset(path_to_dataset, 'test', config)

    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    depth_model = model.DepthCNN(config)
    depth_model.cuda()

    #checkpoint_dir = 'checkpoints/nyudepth20200115T1616/mask_rcnn_nyudepth_0050.pth'
    #checkpoint_dir = 'checkpoints/nyudepth20191207T1213/mask_rcnn_nyudepth_0200.pth'
    checkpoint_dir = 'checkpoints/nyudepth20200117T0105/mask_rcnn_nyudepth_0050.pth'
    #checkpoint_dir = 'checkpoints/nyudepth20191230T1055/mask_rcnn_nyudepth_0200.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))

    errors = []
    chamfer_scores = []
    step = 0

    depth_model.eval()
    with torch.no_grad():

        for i in range(len(dataset_val)):

            inputs = dataset_val[i]
            rgb = inputs[0]
            #rgb = rgb.transpose(1, 2, 0)
            #rgb = mold_image(rgb, config)
            #rgb = rgb.transpose(2, 0, 1)
            images = torch.from_numpy(rgb).unsqueeze(0)
            gt_depths = torch.from_numpy(inputs[2]).unsqueeze(0)

            # Wrap in variables
            images = Variable(images)
            gt_depths = Variable(gt_depths)

            images = images.cuda().float()
            gt_depths = gt_depths.cuda().float()

            depth_np = depth_model.predict([images, gt_depths], mode='inference')

            #print("pred: ", depth_np[0].shape)
            #print("gt: ", gt_depths.shape)
            #print("pred type:", depth_np[0].dtype)

            cham = calculate_chamfer_scene(gt_depths[:, 80:560], depth_np[0][:, 80:560].cuda())
            chamfer_scores.append(cham)

            #loss = CalculateLosses(depth_np[0][0, 80:560], gt_depths[0, 80:560, :])
            #losses.append(loss)

            depth_pred = depth_np[0][0, 80:560, :].detach().cpu().numpy()
            depth_gt = gt_depths[0, 80:560, :].cpu().numpy()

            err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
            #err = compute_errors(depth_gt, depth_pred)
            errors.append(err)

            #err2 = compute_errors_nomask(depth_gt, depth_pred)
            #errors2.append(err2)

            step += 1

            if step % 100 == 0:
                print(" HERE: ", step)

    e = np.array(errors).mean(0).tolist()
    #print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse','rmse_log', 'a1', 'a2', 'a3'))
    #print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],e[4], e[5], e[6], e[7]))

    #rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[3], e[4], e[5], e[6], e[7]))


    c = np.array(chamfer_scores).mean(0)
    print("Chamfer score: ", c)


def evaluate_roiregions():

    config = nyu.NYUConfig()
    path_to_dataset = '../NYU_data'

    dataset_val = nyu.NYUDataset(path_to_dataset, 'test', config)

    #test_generator = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1)
    #test_iter = iter(test_generator)

    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    depth_model = model.DepthCNN(config)
    depth_model.cuda()

    checkpoint_dir = 'checkpoints/nyudepth20200112T1758/mask_rcnn_nyudepth_0200.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))

    loss_roi = []
    with torch.no_grad():
        for i in range(len(dataset_val)):

            #data_item = next(test_iter)
            #inputs = data_item
            inputs = dataset_val[i]
            images = inputs[0]
            # image_metas = inputs[1]
            # rpn_match = inputs[2]
            # rpn_bbox = inputs[3]
            # gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_depth = inputs[7]

            from torch.autograd import Variable

            # Wrap in variables
            images = Variable(images)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            gt_depth = Variable(gt_depth)

            images = images.cuda()
            gt_boxes = gt_boxes.cuda()
            gt_masks = gt_masks.cuda()
            gt_depth = gt_depth.cuda()

            depth_np = depth_model.predict([images, gt_depth], mode='inference')

            gt_dep = gt_depth.detach().cpu().numpy()[0]
            pred_dep = depth_np[0].detach().cpu().numpy()[0]
            masks = gt_masks.detach().cpu().numpy()[0]

            roi_err = eval_roi_accuracy(gt_dep, pred_dep, masks)
            loss_roi.append(roi_err)

            if i % 50 == 0:
                print(i)

    e = np.array(loss_roi).mean(0).tolist()
    print("Masked regions depth mean: ", e)


if __name__ == '__main__':
    augmentation = iaa.Sometimes(.667, iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.25.
        # But we only blur about 50% of all images.
        # iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.25))),
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

    # augmentation = iaa.Sometimes(.667, iaa.Fliplr(0.5))  # horizontal flips)  # apply augmenters in random order

    #augmentation = iaa.Fliplr(0.5)

    #train_maskrcnn(augmentation=augmentation, depth_weight=10)

    #train_depth(augmentation=augmentation)

    #train_depth()
    #evaluate_roiregions()
    evaluate_solodepth()
    #evaluate_roiregions()


    #train_depthrcnn(augmentation=augmentation)