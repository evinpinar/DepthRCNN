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



def train_maskrcnn(augmentation=None):
	config = nyu.NYUConfig()
	path_to_dataset = '../NYU_data'

	standard_split = True

	if standard_split:
		dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
		dataset_val = nyu.NYUDataset(path_to_dataset, 'test', config)
	else:
		dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
		dataset_val = nyu.NYUDataset(path_to_dataset, 'val', config)
		dataset_test = nyu.NYUDataset(path_to_dataset, 'test', config)

	mask_model = model.MaskRCNN(config)
	mask_model.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	mask_model.load_weights(resnet_path)

	#checkpoint_dir = 'checkpoints/nyudepth20190712T1658/mask_rcnn_nyudepth_0100.pth'
	#mask_model.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 120
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 20

	epochs = 300
	layers = "5+"  # options: 3+, 4+, 5+, heads, all

	depth_weight = 5

	mask_model.train()
	start = timer()
	mask_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs,
						   layers=layers, depth_weight=depth_weight)

	end = timer()
	print('Total training time: ', end - start)


def train_depth(augmentation=None):
	config = nyu.NYUConfig()
	path_to_dataset = '../NYU_data'

	dataset_train = nyu.NYUDepthDataset(path_to_dataset, 'train', config, augment=False, augmentation=augmentation)
	dataset_val = nyu.NYUDepthDataset(path_to_dataset, 'test', config)
	#dataset_test = nyu.NYUDepthDataset(path_to_dataset, 'test', config)

	depth_model = model.DepthCNN(config)
	depth_model.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	depth_model.load_weights(resnet_path)

	#checkpoint_dir = 'test/nyudepth20190710T1750/mask_rcnn_nyudepth_0300.pth'
	#depth_model.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 1000
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 100

	epochs = 300

	depth_model.train()
	start = timer()
	depth_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)
	end = timer()
	print('Total training time: ', end - start)

	## Modify this to see graph on tensorboard!
	# out = depth_model(dataset_test[])
	# writer = SummaryWriter()
	# writer.add_graph(model, out)
	# writer.close()

def train_depthmask():
	config = nyu.NYUConfig()
	path_to_dataset = '../NYU_data'

	dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config)
	dataset_test = nyu.NYUDataset(path_to_dataset, 'test', config)

	model_maskdepth = MaskDepthRCNN(config)
	model_maskdepth.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	model_maskdepth.load_weights(resnet_path)

	# checkpoint_dir = 'test/nyudepth20190711T0843/mask_rcnn_nyudepth_0100.pth'
	# mask_model.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 50
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 20

	epochs = 100
	layers = "5+"  # options: 3+, 4+, 5+, heads, all

	model_maskdepth.train()
	start = timer()
	model_maskdepth.train_model(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=epochs,
						   layers=layers)
	end = timer()
	print('Total training time: ', end - start)

if __name__ == '__main__':

	augmentation = iaa.Sometimes(.667, iaa.Sequential([
		iaa.Fliplr(0.5),  # horizontal flips
		iaa.Crop(percent=(0, 0.1)),  # random crops
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
		iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
			# translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
			rotate=(-180, 180),
			# shear=(-8, 8)
		)
	], random_order=True))  # apply augmenters in random order

	train_maskrcnn(augmentation=augmentation)