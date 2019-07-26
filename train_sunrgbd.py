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
import sun
from models.model_maskdepthrcnn import *
from timeit import default_timer as timer
from config import Config
import utils
from tensorboardX import SummaryWriter
import imgaug.augmenters as iaa


def train_model(mask_model, config, train_dataset, val_dataset, learning_rate, epochs, layers, depth_weight=0):

	layer_regex = {
		# all layers but the backbone
		"heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(depth.*)",
		# From a specific Resnet stage and up
		"3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(depth.*)",
		"4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(depth.*)",
		"5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)|(depth.*)",
		# All layers
		"all": ".*",
	}
	if layers in layer_regex.keys():
		layers = layer_regex[layers]

	# Train
	modellib.log("\nStarting at epoch {}. LR={}\n".format(epochs + 1, config.LEARNING_RATE))
	modellib.log("Checkpoint Path: {}".format(mask_model.checkpoint_path))
	mask_model.set_trainable(layers)

	# Optimizer object
	# Add L2 Regularization
	# Skip gamma and beta weights of batch normalization layers.
	trainables_wo_bn = [param for name, param in mask_model.named_parameters() if
						param.requires_grad and not 'bn' in name]
	trainables_only_bn = [param for name, param in mask_model.named_parameters() if
						  param.requires_grad and 'bn' in name]

	import torch.optim as optim

	optimizer = optim.SGD([
		{'params': trainables_wo_bn, 'weight_decay': config.WEIGHT_DECAY},
		{'params': trainables_only_bn}
	], lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM)

	optimizer.zero_grad()

	train_generator = modellib.data_generator(train_dataset, config, shuffle=True, augment=True, batch_size=1)
	val_generator = modellib.data_generator(val_dataset, config, shuffle=True, augment=True, batch_size=1)

	for epoch in range(0, epochs + 1):
		modellib.log("Epoch {}/{}.".format(epoch, epochs))

		# Training
		loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth = mask_model.train_epoch(
			train_generator, optimizer, config.STEPS_PER_EPOCH, depth_weight)

		# Validation
		val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_depth = mask_model.valid_epoch(
			val_generator, mask_model.config.VALIDATION_STEPS, depth_weight)

		# Statistics
		mask_model.loss_history.append(
			[loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth])
		mask_model.val_loss_history.append(
			[val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
			 val_loss_mrcnn_mask, val_loss_depth])
		visualize.plot_loss(mask_model.loss_history, mask_model.val_loss_history, save=True, log_dir=mask_model.log_dir)

		if not os.path.exists(mask_model.log_dir):
			os.makedirs(mask_model.log_dir)



def train_maskrcnn(augmentation=None, depth_weight=0):

	config = sun.SunConfig()
	SUN_DIR = '../SUNRGBD/train'

	dataset_train = sun.SunDataset()
	dataset_train.load_sun(SUN_DIR, "train")
	dataset_train.prepare()

	dataset_val = sun.SunDataset()
	dataset_val.load_sun(SUN_DIR, "val")
	dataset_val.prepare()

	mask_model = modellib.MaskRCNN(config)
	mask_model.cuda()

	#resnet_path = '../resnet50_imagenet.pth'
	#mask_model.load_weights(resnet_path)

	checkpoint_dir = 'checkpoints/sun20190726T1130/mask_rcnn_sun_0005.pth'
	mask_model.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 2000
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 200

	depth_weight = 0
	config.PREDICT_DEPTH = False

	epochs = 50
	layers = "heads"  # options: 3+, 4+, 5+, heads, all

	start = timer()
	mask_model.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
						   layers=layers, depth_weight=depth_weight)

	end = timer()
	print('Total training time: ', end - start)


if __name__ == '__main__':

	import warnings
	#warnings.filterwarnings("ignore")
	print("starting!")

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		train_maskrcnn(augmentation=None, depth_weight=0)

