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

from evaluate_utils import *

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


	config.STEPS_PER_EPOCH = 2000
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 200

	config.PREDICT_DEPTH = True

	epochs = 50
	layers = "heads"  # options: 3+, 4+, 5+, heads, all

	mask_model = modellib.MaskRCNN(config)
	mask_model.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	mask_model.load_weights(resnet_path)

	# checkpoint_dir = 'checkpoints/sun20190726T1130/mask_rcnn_sun_0005.pth'
	# mask_model.load_state_dict(torch.load(checkpoint_dir))

	start = timer()
	mask_model.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
						   layers=layers, depth_weight=depth_weight)

	end = timer()
	print('Total training time: ', end - start)


def train_roidepth(augmentation=None, depth_weight=1):

	config = sun.SunConfig()
	SUN_DIR = '../SUNRGBD/train'

	dataset_train = sun.SunDataset()
	dataset_train.load_sun(SUN_DIR, "train")
	dataset_train.prepare()

	dataset_val = sun.SunDataset()
	dataset_val.load_sun(SUN_DIR, "val")
	dataset_val.prepare()

	config.STEPS_PER_EPOCH = 2000
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 200

	config.PREDICT_DEPTH = True

	epochs = 50
	layers = "heads"  # options: 3+, 4+, 5+, heads, all

	model_maskdepth = MaskDepthRCNN(config)
	model_maskdepth.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	model_maskdepth.load_weights(resnet_path)

	#checkpoint_dir = 'checkpoints/nyudepth20190721T1816/mask_depth_rcnn_nyudepth_0025.pth'
	#model_maskdepth.load_state_dict(torch.load(checkpoint_dir))


	start = timer()
	model_maskdepth.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
								layers=layers, depth_weight=depth_weight)
	end = timer()
	print('Total training time: ', end - start)


def train_solodepth(augmentation=None):
	config = sun.SunConfig()
	SUN_DIR = '../SUNRGBD/train'

	dataset_train = sun.SunDataset()
	dataset_train.load_sun(SUN_DIR, "train")
	dataset_train.prepare()

	dataset_val = sun.SunDataset()
	dataset_val.load_sun(SUN_DIR, "val")
	dataset_val.prepare()

	depth_model = modellib.DepthCNN(config)
	depth_model.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	depth_model.load_weights(resnet_path)

	#checkpoint_dir = 'checkpoints/sun20190726T1130/mask_rcnn_sun_0005.pth'
	#depth_model.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 2000
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 200

	config.PREDICT_DEPTH = False

	epochs = 50
	layers = "heads"  # options: 3+, 4+, 5+, heads, all

	start = timer()
	depth_model.train_model2(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
							layers=layers)

	end = timer()
	print('Total training time: ', end - start)


def evaluate():
	config = sun.SunConfig()
	SUN_DIR_test = '../SUNRGBD'

	dataset_test = sun.SunDataset()
	dataset_test.load_sun(SUN_DIR_test, "test")
	dataset_test.prepare()

	print("--TEST--")
	print("Image Count: {}".format(len(dataset_test.image_ids)))
	print("Class Count: {}".format(dataset_test.num_classes))
	for i, info in enumerate(dataset_test.class_info):
		print("{:3}. {:50}".format(i, info['name']))

	depth_model = modellib.DepthCNN(config)
	depth_model.cuda()

	checkpoint_dir = 'checkpoints/sun20190729T1024/mask_rcnn_sun_0050.pth'
	depth_model.load_state_dict(torch.load(checkpoint_dir))

	test_datagenerator = modellib.data_generator_onlydepth(dataset_test, config, shuffle=True, augment=False, batch_size=1)
	errors = np.zeros(8)
	step = 0
	steps = 5000
	for inputs in test_datagenerator:
		images = inputs[0]
		gt_depths = inputs[2]

		# Wrap in variables
		images = Variable(images)
		gt_depths = Variable(gt_depths)

		images = images.cuda()
		gt_depths = gt_depths.cuda()

		depth_np = depth_model.predict([images, gt_depths], mode='inference')

		depth_pred = depth_np[0][0, 80:560, :].detach().cpu().numpy()
		depth_gt = gt_depths[0, 80:560, :].cpu().numpy()

		err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
		errors = errors + err
		step += 1

		if step %100 == 0:
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

	import warnings
	#warnings.filterwarnings("ignore")
	print("starting!")

	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		#train_solodepth(augmentation=None)
		#evaluate()
		train_maskrcnn(depth_weight=5)
		#train_roidepth()

