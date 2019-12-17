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
	config.BATCH_SIZE = 5

	config.DEPTH_THRESHOLD = 0
	config.PREDICT_DEPTH = True
	config.DEPTH_LOSS = 'L2' # Options: L1, L2, BERHU
	config.CHAM_LOSS = False
	config.GRAD_LOSS = False

	depth_model = model.DepthCNN(config)

	depth_model.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	depth_model.load_weights(resnet_path)

	#checkpoint_dir = 'checkpoints/nyudepth20191003T1927/mask_rcnn_nyudepth_0100.pth'
	#depth_model.load_state_dict(torch.load(checkpoint_dir))

	depth_model.train()
	start = timer()

	epochs = 100
	depth_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)

	config.LEARNING_RATE /= 10
	epochs = 200
	depth_model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)


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

	depth_model.train()
	start = timer()

	epochs = 100
	depth_model.train_model4(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)

	config.LEARNING_RATE /= 10
	epochs = 200
	depth_model.train_model4(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=epochs)


	end = timer()
	print('Total training time: ', end - start)


def train_depthrcnn(augmentation=None, depth_weight=0):
	config = nyu.NYUConfig()
	path_to_dataset = '../NYU_data'

	dataset_train = nyu.NYUDataset(path_to_dataset, 'train', config, augmentation=augmentation)
	dataset_test = nyu.NYUDataset(path_to_dataset, 'test', config, augmentation=augmentation)

	model_maskdepth = MaskDepthRCNN(config)
	model_maskdepth.cuda()

	resnet_path = '../resnet50_imagenet.pth'
	model_maskdepth.load_weights(resnet_path)

	#checkpoint_dir = 'checkpoints/nyudepth20190721T1816/mask_depth_rcnn_nyudepth_0025.pth'
	#model_maskdepth.load_state_dict(torch.load(checkpoint_dir))

	config.STEPS_PER_EPOCH = 300
	config.TRAIN_ROIS_PER_IMAGE = 100
	config.VALIDATION_STEPS = 25

	epochs = 100
	layers = "heads"  # options: 3+, 4+, 5+, heads, all

	start = timer()
	model_maskdepth.train_model(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=epochs,
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

	checkpoint_dir = 'checkpoints/nyudepth20191207T1216/mask_rcnn_nyudepth_0200.pth'
	#checkpoint_dir = 'checkpoints/nyudepth20190817T0911/mask_rcnn_nyudepth_0200.pth'
	depth_model.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))

	errors = []
	chamfer_scores = []
	step = 0

	depth_model.eval()
	with torch.no_grad():

		for i in range(len(dataset_val)):

			inputs = dataset_val[i]
			images = torch.from_numpy(inputs[0]).unsqueeze(0)
			gt_depths = torch.from_numpy(inputs[2]).unsqueeze(0)

			# Wrap in variables
			images = Variable(images)
			gt_depths = Variable(gt_depths)

			images = images.cuda()
			gt_depths = gt_depths.cuda()

			depth_np = depth_model.predict([images, gt_depths], mode='inference')

			#print("pred: ", depth_np[0].shape)
			#print("gt: ", gt_depths.shape)
			#print("pred type:", depth_np[0].dtype)

			cham = calculate_chamfer_scene(gt_depths[:, 80:560], depth_np[0][:, 80:560].cuda())
			chamfer_scores.append(cham)

			depth_pred = depth_np[0][0, 80:560, :].detach().cpu().numpy()
			depth_gt = gt_depths[0, 80:560, :].cpu().numpy()

			err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
			#err = compute_errors(depth_gt, depth_pred)
			errors.append(err)


			step += 1

			if step % 100 == 0:
				print(" HERE: ", step)

	e = np.array(errors).mean(0).tolist()
	#print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse','rmse_log', 'a1', 'a2', 'a3'))
	#print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],e[4], e[5], e[6], e[7]))


	print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'))
	print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5], e[6]))

	c = np.array(chamfer_scores).mean(0)
	print("Chamfer score: ", c)



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
	#train_depth_cham_masked()
	evaluate_solodepth()