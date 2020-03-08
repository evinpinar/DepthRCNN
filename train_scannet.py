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

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}

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

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}


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


# Train for shift heads and roi depths, no attention, no shift
def train_roidepth_base(augmentation=None, depth_weight=1):

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

    config.PREDICT_SHIFT = False
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

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)


    start = timer()

    epochs = 10
    layers = "depth"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/5, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight, augmentation=augmentation, shift_weight=0)

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


# Train with scale head
def train_roidepth_scale(augmentation=None):
    print("ROI training!")

    config = scannet.ScannetConfig()
    scannet_data = '../data/SCANNET/'

    dataset_train = scannet.ScannetDataset()
    dataset_train.load_scannet("train", scannet_data)
    dataset_train.prepare()

    dataset_val = scannet.ScannetDataset()
    dataset_val.load_scannet("val", scannet_data)
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

    config.DEPTH_ATTENTION = False
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = False
    config.DEPTH_SCALE = True

    model_maskdepth = MaskDepthRCNN(config, model_dir='../data/checkpoints')
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    # model_maskdepth.load_weights(resnet_path)
    # model_maskdepth.load_weights(coco_path, iscoco=True)

    # Call checkpoint below in train_model call!

    # model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    ### Load maskrcnn weights
    checkpoint_dir = '../data/checkpoints/scannet20200124T1510/mask_depth_rcnn_scannet_0010.pth'
    checkpoint = torch.load(checkpoint_dir)

    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                  'depth' not in k and 'depthmask' not in k and 'depth_ref' not in k and 'depthmask' not in k}

    state = model_maskdepth.state_dict()
    state.update(state_dict)
    model_maskdepth.load_state_dict(state, strict=False)

    ### Load global depth weights
    depth_checkpoint_dir = '../data/checkpoints/scannet20200202T1605/mask_rcnn_scannet_0015.pth'
    depth_checkpoint = torch.load(depth_checkpoint_dir)

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)

    start = timer()

    epochs = 10
    layers = "depth"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 5, epochs=epochs,
                                layers=layers, depth_weight=depth_weight, augmentation=augmentation, shift_weight=0)

    # model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/2, epochs=epochs,
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
    print('Total training time: ', (end - start) / 60)


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


def train_maskrcnn(augmentation=None):
    print("ROI training!")

    config = scannet.ScannetConfig()
    scannet_data = '../data/SCANNET/'

    dataset_train = scannet.ScannetDataset()
    dataset_train.load_scannet("train", scannet_data)
    dataset_train.prepare()

    dataset_val = scannet.ScannetDataset()
    dataset_val.load_scannet("val", scannet_data)
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

    config.DEPTH_ATTENTION = False
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = False
    config.DEPTH_SCALE = False

    model_maskdepth = MaskDepthRCNN(config, model_dir='../data/checkpoints')
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    # model_maskdepth.load_weights(resnet_path)
    # model_maskdepth.load_weights(coco_path, iscoco=True)

    # Call checkpoint below in train_model call!

    # model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

    ### Load maskrcnn weights
    checkpoint_dir = '../data/checkpoints/scannet20200124T1510/mask_depth_rcnn_scannet_0010.pth'
    checkpoint = torch.load(checkpoint_dir)

    state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                  'depth' not in k and 'depthmask' not in k and 'depth_ref' not in k and 'depthmask' not in k}

    state = model_maskdepth.state_dict()
    state.update(state_dict)
    model_maskdepth.load_state_dict(state, strict=False)

    depth_checkpoint_dir = '../data/checkpoints/scannet20200202T1605/mask_rcnn_scannet_0015.pth'
    depth_checkpoint = torch.load(depth_checkpoint_dir)

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)

    start = timer()

    epochs = 10
    layers = "depthmaskrcnn"  # options: 3+, 4+, 5+, heads, all, depth
    model_maskdepth.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/2, epochs=epochs,
                                layers=layers, depth_weight=depth_weight, augmentation=augmentation)

    # model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE/2, epochs=epochs,
    #                             layers=layers, depth_weight=depth_weight, augmentation=augmentation,
    #                             checkpoint_dir_prev=checkpoint_dir, continue_train=False)

    '''
    epochs = 40
    layers = "all"  # options: 3+, 4+, 5+, heads, all
    model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
                                 layers=layers, depth_weight=depth_weight,
                                 augmentation=augmentation, continue_train=True)

    '''

    end = timer()
    print('Total training time: ', (end - start) / 60)


def evaluate_initial_backbone():
    print("Test the depthrcnn backbone depth to see if depth backbone is effected !")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("val")  # change the number of steps as well
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))

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

    config.DEPTH_ATTENTION = False
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = False

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    resnet_path = '../resnet50_imagenet.pth'
    coco_path = '../mask_rcnn_coco.pth'  # This is based on resnet 101

    # model_maskdepth.load_weights(resnet_path)
    # model_maskdepth.load_weights(coco_path, iscoco=True)

    # Call checkpoint below in train_model call!

    # model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

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

    state_dict = {k: v for k, v in depth_checkpoint.items() if 'depth' in k}

    model_maskdepth.load_state_dict(state_dict, strict=False)

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436
    errors = []
    masked_errs = []
    gt_masked_errs = []
    num_pixels = []
    masked_pixels = []
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

            detections, mrcnn_mask, mrcnn_depth, mrcnn_normals, global_depth, mrcnn_shifts = \
                model_maskdepth.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
                                         mode='inference')

            depth_pred = global_depth.detach().cpu().numpy()[0, 80:560, :]
            depth_gt = gt_depth.cpu().numpy()[0, 80:560, :]

            # Evaluate Global Depth
            err = evaluateDepthsTrue(depth_pred, depth_gt, printInfo=False)
            errors.append(err[:-1])
            num_pixels.append(err[-1])

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

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print("\nTraining code : ")
    print("====== 1. Global depth error: ")
    calculateTrueErrors(errors, num_pixels)

    print("====== 2. GT Masked error: ")
    calculateTrueErrors(gt_masked_errs, gt_masked_pixels)


# depth rcnn with no shift/scale/attn
def evaluate_naive_model():

    print("Evaluate naive depth rcnn!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test", scannet_data="../data/SCANNET/")
    dataset_test.prepare()

    print("--TEST--")
    print("Image Count: {}".format(len(dataset_test.image_ids)))
    print("Class Count: {}".format(dataset_test.num_classes))

    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

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

    config.DEPTH_ATTENTION = False
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = False
    config.DEPTH_SCALE = False

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = '../data/checkpoints/scannet20200227T1132/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0')['model_state_dict'])

    checkpoint_dir = 'checkpoints/scannet20200207T1619/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    step = 0
    steps = 5435

    errors = {i: [] for i in range(1, 8)}
    pixels = {i: [] for i in range(1, 8)}

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
                    model_maskdepth.predict(
                        [images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals],
                        mode='training')

                ### Evaluate Local Depths
                # select nonzero values, class ids
                positive_ix = torch.nonzero(target_class_ids[0] > 0)[:, 0]
                positive_class_ids = target_class_ids[0, positive_ix.data].long()
                indices = torch.stack((positive_ix, positive_class_ids), dim=1)

                ## Gather the depths (predicted and true) that contribute to loss
                target_depths = target_depths.cpu().numpy()
                mrcnn_depths = mrcnn_depths.cpu().numpy()

                true_depth = target_depths[0, indices[:, 0].data, :, :]
                pred_depth = mrcnn_depths[0, indices[:, 0].data, indices[:, 1].data, :, :]

                ### Evaluate roi depths
                roi_err, roi_pix = evaluateRoiDepths(pred_depth, true_depth)
                errors[1] += roi_err
                pixels[1] += roi_pix

                imgs = images.permute(0, 2, 3, 1)

                results = model_maskdepth.detect(imgs, mold_image=False, image_metas=image_metas)
                r = results[0]

                pred_boxes = r['rois']
                pred_masks = r['masks']
                pred_scores = r['scores']
                pred_depths = r['depths']
                pred_glob_depth = r['glob_depth']
                pred_class_ids = r['class_ids']
                gt_depth_whole = gt_depth[0].cpu().numpy()

                exclude_ix = []
                for i in range(pred_boxes.shape[0]):
                    if pred_masks[:, :, i].sum() == 0:
                        exclude_ix.append(i)

                if len(exclude_ix) > 0:
                    pred_class_ids = np.delete(pred_class_ids, exclude_ix, axis=0)
                    pred_masks = np.delete(pred_masks, exclude_ix, axis=2)
                    pred_depths = np.delete(pred_depths, exclude_ix, axis=2)


                n_rois = pred_class_ids.shape[0]

                full_roi_depths = pred_glob_depth.copy()
                # replace them on global -> highest scores remain
                for j in range(n_rois):
                    i = n_rois - j - 1
                    depth = pred_depths[:, :, i] * pred_masks[:, :, i]
                    idx = depth.nonzero()
                    # print(idx.shape)
                    if len(idx) != 0:
                        full_roi_depths[idx] = depth[idx]

                err_global = evaluateDepthsTrue(pred_glob_depth[80:560], gt_depth_whole[80:560], printInfo=False)
                errors[2].append(err_global[:-1])
                pixels[2].append(err_global[-1])

                err_reshifted = evaluateDepthsTrue(full_roi_depths[80:560], gt_depth_whole[80:560], printInfo=False)
                errors[3].append(err_reshifted[:-1])
                pixels[3].append(err_reshifted[-1])


                # Evaluate masked regions [pred masks]
                masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560],
                                                               pred_masks)
                errors[4] += masked_err
                pixels[4] += masked_pix

                masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560],
                                                               pred_masks)
                errors[5] += masked_err
                pixels[5] += masked_pix

                # Evaluate masked regions [gt masks]

                gt_boxes = gt_boxes.data.cpu().numpy()
                gt_class_ids = gt_class_ids.data.cpu().numpy()
                gt_masks = gt_masks.data.cpu().numpy()
                gt_boxes = trim_zeros(gt_boxes[0])
                gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
                # Expand the masks from 56x56 to 640x640
                expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

                masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560],
                                                               expanded_mask)
                errors[6] += masked_err
                pixels[6] += masked_pix

                masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560],
                                                               expanded_mask)
                errors[7] += masked_err
                pixels[7] += masked_pix



                step += 1
                if step % 100 == 0:
                    print(" HERE: ", step)

                # Break after 'steps' steps
                if step == steps - 1:
                    break


    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print("====== 1. ROI accuracy: ")
    calculateTrueErrors(errors[1], pixels[1])

    print("====== 2. Global pred whole accuracy: ")
    calculateTrueErrors(errors[2], pixels[2])

    print("====== 3. Global + ROI whole accuracy: ")
    calculateTrueErrors(errors[3], pixels[3])

    print("====== 4. Global pred GT-masked accuracy: ")
    calculateTrueErrors(errors[4], pixels[4])

    print("====== 5. Global+ROI GT-masked accuracy: ")
    calculateTrueErrors(errors[5], pixels[5])

    print("====== 6. Global pred pred-masked accuracy: ")
    calculateTrueErrors(errors[6], pixels[6])

    print("====== 7. Global+ROI pred-masked accuracy: ")
    calculateTrueErrors(errors[7], pixels[7])


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

            depth_pred = depth_np[0][0, 80:560, :].detach().cpu().numpy()
            depth_gt = gt_depths[0, 80:560, :].cpu().numpy()

            err = evaluateDepthsTrue(depth_pred[80:560], depth_gt[80:560], printInfo=False)
            errors.append(err[:-1])
            num_pixels.append(err[-1])

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

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print("\nTraining code : ")
    print("====== 1. Global depth error: ")
    calculateTrueErrors(errors, num_pixels)

    print("====== 2. GT Masked error: ")
    calculateTrueErrors(gt_masked_errs, gt_masked_pixels)


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

    config.PREDICT_SHIFT = True
    config.DEPTH_ATTENTION = False

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20200207T1619/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436

    errors = {i:[] for i in range(1,15)}
    pixels = {i:[] for i in range(1,15)}

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
            roi_err, roi_pix = evaluateRoiDepths(pred_depth, true_depth)
            errors[1] += roi_err
            pixels[1] += roi_pix

            ### Evaluate shifts ( amount of shift, absrel, rmse )
            shift_err = evaluateDepthsTrue(pred_shift, true_shift)
            rel, relsqr, rmse, rmse_log = shift_err[0], shift_err[1], shift_err[2], shift_err[4]
            #errors[2].append([rel, relsqr, rmse, rmse_log])
            errors[2].append(shift_err[:-1])
            pixels[2].append(shift_err[-1]) # number of shifts(ROIs) for one instance


            ### Evaluate local+global by predicted SHIFT, pred boxes
            # We need the unnormalized box coordinates for that, which comes from inference code.

            imgs = images.permute(0, 2, 3, 1)

            results = model_maskdepth.detect(imgs, mold_image=False, image_metas=image_metas)
            r = results[0]

            pred_boxes = r['rois']
            pred_masks = r['masks']
            pred_scores = r['scores']
            pred_depths = r['depths']
            pred_glob_depth = r['glob_depth']
            pred_class_ids = r['class_ids']
            pred_shifts = r['shifts']

            exclude_ix = []
            for i in range(pred_boxes.shape[0]):
                if pred_masks[:, :, i].sum() == 0:
                    exclude_ix.append(i)

            if len(exclude_ix) > 0:
                pred_boxes = np.delete(pred_boxes, exclude_ix, axis=0)
                pred_class_ids = np.delete(pred_class_ids, exclude_ix, axis=0)
                pred_shifts = np.delete(pred_shifts, exclude_ix, axis=0)
                pred_scores = np.delete(pred_scores, exclude_ix, axis=0)
                pred_masks = np.delete(pred_masks, exclude_ix, axis=2)
                pred_depths = np.delete(pred_depths, exclude_ix, axis=2)

            n_rois = pred_class_ids.shape[0]

            ### Evaluate roi depths

            # Extract the same regions on gt_depth and pred global depth, mask them, calculate shift and apply shift.
            # Pred shift and pred mask used

            gt_depth_rois = np.zeros([640, 640, n_rois])
            gt_shifts = np.zeros([n_rois])
            global_depth_rois = np.zeros([640, 640, n_rois])

            gt_depth_whole = gt_depth[0].cpu().numpy()

            thresh = 0.2
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = gt_depth_whole.copy() * mask
                if depth_area.sum() == 0:
                    gt_shifts[i] = 0
                    gt_depth_rois[:, :, i] = depth_area
                    break
                min_val = depth_area[depth_area > thresh].min()
                gt_shifts[i] = min_val
                depth_area -= pred_shifts[i] #min_val
                gt_depth_rois[:, :, i] = depth_area * mask

            # Find global depth local areas, shifted, masked
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = pred_glob_depth.copy() * mask
                depth_area -= pred_shifts[i]  # gt_shifts[i]
                global_depth_rois[:, :, i] = depth_area * mask

            # Compare roi depth prediction between local vs global depth

            # evaluate errors between rois -> pred rois vs gt rois
            masked_err, masked_pix = evaluateMaskedROIRegions(pred_depths, gt_depth_rois)
            errors[3] += masked_err
            pixels[3] += masked_pix

            # evaluate errors between rois -> crop rois from pred global vs gt rois
            masked_err, masked_pix = evaluateMaskedROIRegions(global_depth_rois, gt_depth_rois)
            errors[4] += masked_err
            pixels[4] += masked_pix

            # Shift prediction error
            err_gt_shift = evaluateDepthsTrue(pred_shifts, gt_shifts)
            errors[5].append(err_gt_shift[:-1])
            pixels[5].append(err_gt_shift[-1])

            # Replace the locals depths on pred global depth by [predicted masks]

            pred_depths_reshifted = np.zeros([640, 640, n_rois])

            # Find gt local areas, shifted, masked
            thresh = 0.2
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth = pred_depths[:, :, i].copy()
                depth_area = depth * mask
                depth_area += pred_shifts[i]  # gt_shifts
                pred_depths_reshifted[:, :, i] = depth_area * mask

            full_roi_depths = pred_glob_depth.copy()
            # resize and replace them on global
            for j in range(n_rois):
                i = n_rois - j - 1
                depth = pred_depths_reshifted[:, :, i] * pred_masks[:, :, i]
                idx = depth.nonzero()
                # print(idx.shape)
                if len(idx) != 0:
                    full_roi_depths[idx] = depth[idx]

            # Evaluate whole depths

            err_reshifted = evaluateDepthsTrue(full_roi_depths[80:560], gt_depth_whole[80:560], printInfo=False)
            errors[6].append(err_reshifted[:-1])
            pixels[6].append(err_reshifted[-1])

            err_global = evaluateDepthsTrue(pred_glob_depth[80:560], gt_depth_whole[80:560], printInfo=False)
            errors[7].append(err_global[:-1])
            pixels[7].append(err_global[-1])

            # Evaluate masked regions [pred masks]
            masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560], pred_masks)
            errors[8] += masked_err
            pixels[8] += masked_pix


            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560], pred_masks)
            errors[9] += masked_err
            pixels[9] += masked_pix

            # Evaluate masked regions [gt masks]

            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_class_ids = gt_class_ids.data.cpu().numpy()
            gt_masks = gt_masks.data.cpu().numpy()
            gt_boxes = trim_zeros(gt_boxes[0])
            gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
            # Expand the masks from 56x56 to 640x640
            expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

            masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560], expanded_mask)
            errors[10] += masked_err
            pixels[10] += masked_pix

            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560], expanded_mask)
            errors[11] += masked_err
            pixels[11] += masked_pix


            ### Evaluate local+global by GT SHIFT, pred boxes
            # a. Whole depth
            # b. Masked regions

            gt_depth_rois = np.zeros([640, 640, n_rois])
            gt_shifts = np.zeros([n_rois])
            global_depth_rois = np.zeros([640, 640, n_rois])

            gt_depth_whole = gt_depth[0].cpu().numpy()

            thresh = 0.2
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = gt_depth_whole.copy() * mask
                if depth_area.sum() == 0:
                    gt_shifts[i] = 0
                    gt_depth_rois[:, :, i] = depth_area
                    break
                min_val = depth_area[depth_area > thresh].min()
                gt_shifts[i] = min_val
                depth_area -= min_val
                gt_depth_rois[:, :, i] = depth_area * mask

            # Find global depth local areas, shifted, masked
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = pred_glob_depth.copy() * mask
                depth_area -= gt_shifts[i]
                global_depth_rois[:, :, i] = depth_area * mask

            pred_depths_reshifted = np.zeros([640, 640, n_rois])

            # Find gt local areas, shifted, masked
            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth = pred_depths[:, :, i].copy()
                depth_area = depth * mask
                depth_area += gt_shifts[i]  # gt_shifts
                pred_depths_reshifted[:, :, i] = depth_area * mask

            full_roi_depths = pred_glob_depth.copy()
            # resize and replace them on global
            for j in range(n_rois):
                i = n_rois - j - 1
                depth = pred_depths_reshifted[:, :, i] * pred_masks[:, :, i]
                idx = depth.nonzero()
                # print(idx.shape)
                if len(idx) != 0:
                    full_roi_depths[idx] = depth[idx]

            # Evaluate whole depth region
            err_reshifted = evaluateDepthsTrue(full_roi_depths[80:560], gt_depth_whole[80:560], printInfo=False)
            errors[12].append(err_reshifted[:-1])
            pixels[12].append(err_reshifted[-1])

            # Evaluate masked regions [pred masks]
            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560], pred_masks)
            errors[13] += masked_err
            pixels[13] += masked_pix

            # Evaluate masked regions [gt masks]
            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560],
                                                           expanded_mask)
            errors[14] += masked_err
            pixels[14] += masked_pix

            step += 1
            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break



    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print("\nTraining code : ")
    print("====== 1. Local depth eval, target vs pred depths: ")
    calculateTrueErrors(errors[1], pixels[1])

    print("====== 2. Shifts eval, target vs pred shifts: ")
    calculateTrueErrors(errors[2], pixels[2])

    print("\nDetection code : ")
    print("====== 3. ROI depths eval: ")
    calculateTrueErrors(errors[3], pixels[3])

    print("====== 4. ROI depth on global pred depth eval: ")
    calculateTrueErrors(errors[4], pixels[4])

    print("====== 5. Pred shifts vs GT shifts(calculated): ")
    calculateTrueErrors(errors[5], pixels[5])

    print("====== 6. Whole depth eval on R + Predshift: ")
    calculateTrueErrors(errors[6], pixels[6])

    print("====== 7. Whole depth eval on pred global: ")
    calculateTrueErrors(errors[7], pixels[7])

    print("====== 8. Pred masked eval on R + Predshift: ")
    calculateTrueErrors(errors[8], pixels[8])

    print("====== 9. Pred masked eval on G: ")
    calculateTrueErrors(errors[9], pixels[9])

    print("====== 10. GT masked eval on R + Predshift: ")
    calculateTrueErrors(errors[10], pixels[10])

    print("====== 11. GT masked eval on G: ")
    calculateTrueErrors(errors[11], pixels[11])

    print("====== 12. Whole depth eval on R + GTshift: ")
    calculateTrueErrors(errors[12], pixels[12])

    print("====== 13. Pred masked eval on R + GTshift: ")
    calculateTrueErrors(errors[13], pixels[13])

    print("====== 14. GT masked eval on R + GTshift: ")
    calculateTrueErrors(errors[14], pixels[14])


def evaluate_scale():
    print("Model evaluation for training set!")

    config = scannet.ScannetConfig()

    dataset_test = scannet.ScannetDataset()
    dataset_test.load_scannet("test", scannet_data = '../data/SCANNET/')  # change the number of steps as well
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

    config.DEPTH_ATTENTION = False
    config.PREDICT_SHIFT = False
    config.ATTENTION_GT_MASK = False
    config.DEPTH_SCALE = True

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = '../data/checkpoints/scannet20200304T1848/mask_depth_rcnn_scannet_0010.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    # test_datagenerator = modellib.data_generator(dataset_test, config, shuffle=False, augment=False, batch_size=1)
    test_generator = data_generator2(dataset_test, config, shuffle=False, augment=False, batch_size=1,
                                     augmentation=None)

    step = 0
    steps = 5436
    # train = 16830, val = 2635, test = 5436

    errors = {i:[] for i in range(1,15)}
    pixels = {i:[] for i in range(1,15)}

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

            ### Evaluate Local Depths
            # select nonzero values, class ids
            positive_ix = torch.nonzero(target_class_ids[0] > 0)[:, 0]
            positive_class_ids = target_class_ids[0, positive_ix.data].long()
            indices = torch.stack((positive_ix, positive_class_ids), dim=1)

            ## Gather the depths (predicted and true) that contribute to loss
            target_depths = target_depths.cpu().numpy()
            mrcnn_depths = mrcnn_depths.cpu().numpy()

            true_depth = target_depths[0, indices[:, 0].data, :, :]
            pred_depth = mrcnn_depths[0, indices[:, 0].data, indices[:, 1].data, :, :]

            ### Evaluate roi depths
            roi_err, roi_pix = evaluateRoiDepths(pred_depth, true_depth)
            errors[1] += roi_err
            pixels[1] += roi_pix

            ### Evaluate local+global by predicted SHIFT, pred boxes
            # We need the unnormalized box coordinates for that, which comes from inference code.

            imgs = images.permute(0, 2, 3, 1)

            results = model_maskdepth.detect(imgs, mold_image=False, image_metas=image_metas)
            r = results[0]

            pred_boxes = r['rois']
            pred_masks = r['masks']
            pred_scores = r['scores']
            pred_depths = r['depths']
            pred_glob_depth = r['glob_depth']
            pred_class_ids = r['class_ids']
            pred_shifts = r['shifts']


            exclude_ix = []
            for i in range(pred_boxes.shape[0]):
                if pred_masks[:, :, i].sum() == 0:
                    exclude_ix.append(i)

            if len(exclude_ix) > 0:
                pred_boxes = np.delete(pred_boxes, exclude_ix, axis=0)
                pred_class_ids = np.delete(pred_class_ids, exclude_ix, axis=0)
                pred_shifts = np.delete(pred_shifts, exclude_ix, axis=0)
                pred_scores = np.delete(pred_scores, exclude_ix, axis=0)
                pred_masks = np.delete(pred_masks, exclude_ix, axis=2)
                pred_depths = np.delete(pred_depths, exclude_ix, axis=2)

            n_rois = pred_class_ids.shape[0]

            ### Evaluate roi depths

            # Extract the same regions on gt_depth and pred global depth, mask them.
            # Scale them to small size.
            # Pred shift and pred mask used.

            if pred_depths.shape[0] == 0: # Resolve the bug
                continue

            mini_shape = 56
            gt_depth_rois = np.zeros([640, 640, n_rois])
            gt_shifts = np.zeros([n_rois])
            global_depth_rois = np.zeros([640, 640, n_rois])
            gt_depth_whole = gt_depth[0].cpu().numpy()

            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = gt_depth_whole.copy() * mask
                if depth_area.sum() == 0:
                    gt_depth_rois[:, :, i] = depth_area
                    break
                y1, x1, y2, x2 = pred_boxes[i][:4]
                # print("y : ", y1, x1, y2, x2)
                delta = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
                delta = int(delta * 640)  # because it was normalized coordinate
                depth_area = depth_area * (delta / mini_shape)
                gt_depth_rois[:, :, i] = depth_area * mask

            for i in range(n_rois):
                mask = pred_masks[:, :, i]
                depth_area = pred_glob_depth.copy() * mask
                y1, x1, y2, x2 = pred_boxes[i][:4]
                delta = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
                delta = int(delta * 640)  # because it was normalized coordinate
                depth_area = depth_area * (delta / mini_shape)
                global_depth_rois[:, :, i] = depth_area * mask

            # print("Pred, globa, gt rois: ", pred_depths.shape, global_depth_rois.shape, gt_depth_rois.shape)
            # Compare roi depth prediction between local vs global depth

            # evaluate errors between rois -> pred rois vs gt rois
            masked_err, masked_pix = evaluateMaskedROIRegions(pred_depths, gt_depth_rois)
            errors[2] += masked_err
            pixels[2] += masked_pix

            # evaluate errors between rois -> crop rois from pred global vs gt rois
            masked_err, masked_pix = evaluateMaskedROIRegions(global_depth_rois, gt_depth_rois)
            errors[3] += masked_err
            pixels[3] += masked_pix

            # Replace the locals depths on pred global depth by [predicted masks]

            pred_depths_rescaled = np.zeros([640, 640, n_rois])

            # Find gt local areas, shifted, masked
            n_rois = pred_scores.shape[0]
            for i in range(n_rois):
                y1, x1, y2, x2 = pred_boxes[i]
                # delta = max(y2 - y1, x2 - x1)
                # delta = x2-x1
                delta = np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))
                pred_depths_rescaled[:, :, i] = pred_depths[:, :, i] * (mini_shape / delta)

            full_roi_depths = pred_glob_depth.copy()
            # resize and replace them on global
            for j in range(n_rois):
                i = n_rois - j - 1
                depth = pred_depths_rescaled[:, :, i] * pred_masks[:, :, i]
                idx = depth.nonzero()
                # print(idx.shape)
                if len(idx) != 0:
                    full_roi_depths[idx] = depth[idx]

            # Evaluate whole depths

            err_reshifted = evaluateDepthsTrue(full_roi_depths[80:560], gt_depth_whole[80:560], printInfo=False)
            errors[4].append(err_reshifted[:-1])
            pixels[4].append(err_reshifted[-1])

            err_global = evaluateDepthsTrue(pred_glob_depth[80:560], gt_depth_whole[80:560], printInfo=False)
            errors[5].append(err_global[:-1])
            pixels[5].append(err_global[-1])

            # Evaluate masked regions [pred masks]
            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560], pred_masks)
            errors[6] += masked_err
            pixels[6] += masked_pix


            masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560], pred_masks)
            errors[7] += masked_err
            pixels[7] += masked_pix

            # Evaluate masked regions [gt masks]

            gt_boxes = gt_boxes.data.cpu().numpy()
            gt_class_ids = gt_class_ids.data.cpu().numpy()
            gt_masks = gt_masks.data.cpu().numpy()
            gt_boxes = trim_zeros(gt_boxes[0])
            gt_masks = gt_masks[0, :gt_boxes.shape[0], :, :]
            # Expand the masks from 56x56 to 640x640
            expanded_mask = utils.expand_mask(gt_boxes, gt_masks.transpose(1, 2, 0), (640, 640, 3))

            masked_err, masked_pix = evaluateMaskedRegions(pred_glob_depth[80:560], gt_depth_whole[80:560], expanded_mask)
            errors[8] += masked_err
            pixels[8] += masked_pix

            masked_err, masked_pix = evaluateMaskedRegions(full_roi_depths[80:560], gt_depth_whole[80:560], expanded_mask)
            errors[9] += masked_err
            pixels[9] += masked_pix


            step += 1
            if step % 100 == 0:
                print(" HERE: ", step)

            # Break after 'steps' steps
            if step == steps - 1:
                break



    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print("\nTraining code : ")
    print("====== 1. Local depth eval, target vs pred depths: ")
    calculateTrueErrors(errors[1], pixels[1])


    print("\nDetection code : ")
    print("====== 2. ROI depths eval: ")
    calculateTrueErrors(errors[2], pixels[2])

    print("====== 3. ROI depth from global pred depth eval: ")
    calculateTrueErrors(errors[3], pixels[3])

    print("====== 4. Whole depth eval on rescaled R + Predshift: ")
    calculateTrueErrors(errors[4], pixels[4])

    print("====== 5. Whole depth eval on pred global G: ")
    calculateTrueErrors(errors[5], pixels[5])

    print("====== 6. Pred masked eval on rescaled R + Predshift: ")
    calculateTrueErrors(errors[6], pixels[6])

    print("====== 7. Pred masked eval on G: ")
    calculateTrueErrors(errors[7], pixels[7])

    print("====== 8. GT masked eval on rescaled R + Predshift: ")
    calculateTrueErrors(errors[8], pixels[8])

    print("====== 9. GT masked eval on G: ")
    calculateTrueErrors(errors[9], pixels[9])


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

        #train_roidepth_scale(augmentation)

        #train_maskrcnn(augmentation)

        evaluate_scale()

        #train_roidepth(augmentation)

        #evaluate_true_attention()

        #evaluate_true_roiregions()

        #evaluate_true_shift()
        #train_roidepth_base(augmentation)

        #count_objectregions()
        #train_roidepth(augmentation)
        #train_solodepth(augmentation)

        #evaluate_roidepth()
        #evaluate_roidepth()



