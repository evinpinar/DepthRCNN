import os
import numpy as np
import torch
import scipy.io
import h5py
import cv2

#from config_original import Config
from config import Config
import utils_original as utils
from torch.utils.data import Dataset
from scipy.ndimage.filters import gaussian_filter

############################################################
#  Configurations
############################################################

class NYUConfig(Config):
	"""Configuration for training on the NYU Depth v2  dataset.
	Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name
	NAME = "nyudepth"

	# Depth Prediction options
	PREDICT_DEPTH = True
	DEPTH_LOSS = 'L1' # Options: L1, L2, BERHU

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 40 + 1  # Background + nyu40 labels #original 894

	TRAIN_ROIS_PER_IMAGE = 200

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9

	#GLOBAL_MASK = False #?

	BILINEAR_UPSAMPLING = True


############################################################
#  Dataset
############################################################

class NYUDataset(Dataset):

	def __init__(self, path_to_dataset, subset, config, augmentation=None):

		self.config = config
		self.augmentation = augmentation

		assert subset in ["train", "test", "val"]
		image_ids = []
		if subset is "train":
			image_ids = np.load(path_to_dataset+'/train_split.npy')
		elif subset is "test":
			image_ids = np.load(path_to_dataset+'/test_split.npy')
		else:
			image_ids = np.load(path_to_dataset + '/val_split.npy')

		print("Extraction from numpy files...")
		imgs = []
		depths = []
		labels = []
		instances = []
		for i in range(len(image_ids)):
			ind = image_ids[i]
			imgs.append(path_to_dataset + "/rgb/" + str(ind))
			depths.append(path_to_dataset + "/depth/" + str(ind))
			labels.append(path_to_dataset + "/label40/" + str(ind))
			instances.append(path_to_dataset + "/instance/" + str(ind))

		self.image_ids = image_ids
		self.images = imgs
		self.depths = depths
		self.labels = labels
		self.instances = instances

		self.class_names = [{"id": 0, "name": "BG"}]

		names = np.load(path_to_dataset + '/names40.npy')
		for i in range(len(names)):
			self.class_names.append({"id": (i+1), "name": names[i]})


		print("Extracting anchors...")
		# Anchors
		# [anchor_count, (y1, x1, y2, x2)]
		self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
													  config.RPN_ANCHOR_RATIOS,
													  config.BACKBONE_SHAPES,
													  config.BACKBONE_STRIDES,
													  config.RPN_ANCHOR_STRIDE)

	def __getitem__(self, i):

		# prepare image [H, W, 3]
		#img = np.transpose(self.images[i], axes=[2, 1, 0])
		# img = img.astype(np.uint8)

		# prepare depth map [H, W]
		# depth = np.transpose(self.depths[i], axes=[1, 0])
		# depth = (depth / depth.max()) * 255
		# depth = depth.astype(np.float32)
		# depth = depth / 4.0  # Normalize

		# extract masks of instances [H, W, N] and labels [N]

		label = np.load(self.labels[i] + ".npy")

		instance = np.load(self.instances[i]+'.npy')
		masks, class_ids = getInstanceMasks(label, instance)

		image = np.load(self.images[i]+'.npy')
		depth = np.load(self.depths[i]+'.npy')/4

		#print(class_ids.shape)

		#print("Loading ground truths...")
		image, image_metas, gt_class_ids, gt_boxes, gt_masks, depth = load_image_gt(self.config, i, image, depth, masks,
																					class_ids, augmentation=self.augmentation)

		## RPN Targets
		#print("Loading RPN targets...")
		rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors, gt_class_ids, gt_boxes, self.config)


		if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
			ids = np.random.choice(
				np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
			gt_class_ids = gt_class_ids[ids]
			gt_boxes = gt_boxes[ids]
			gt_masks = gt_masks[:, :, ids]

		rpn_match = rpn_match[:, np.newaxis]
		image = utils.mold_image(image.astype(np.float32), self.config)

		depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)
		#segmentation = np.concatenate([np.full((80, 640), fill_value=-1, dtype=np.int32), label,
									   #np.full((80, 640), fill_value=-1, dtype=np.int32)], axis=0)

		info = [image.transpose((2, 0, 1)).astype(np.float32), image_metas, rpn_match, rpn_bbox.astype(np.float32),
				gt_class_ids, gt_boxes.astype(np.float32), gt_masks.transpose((2, 0, 1)).astype(np.float32),
				depth.astype(np.float32)]

		return info

	def __len__(self):
		return len(self.image_ids)


class NYUDepthDataset(Dataset):

	def __init__(self, path_to_dataset, subset, config, augment=False, augmentation=None):

		self.config = config
		self.augment = augment
		self.augmentation = augmentation
		self.transform = False

		assert subset in ["train", "test", "val"]
		image_ids = []
		if subset is "train":
			image_ids = np.load(path_to_dataset+'/train_split_real.npy')
		elif subset is "test":
			image_ids = np.load(path_to_dataset+'/test_split_real.npy')
			self.transform = True
		else:
			image_ids = np.load(path_to_dataset + '/val_split.npy')

		print("Extraction from numpy files...")
		imgs = []
		depths = []
		if subset is "test":
			for i in range(len(image_ids)):
				ind = image_ids[i]
				imgs.append(path_to_dataset + "/rgb_test_cropped/" + str(ind))
				depths.append(path_to_dataset + "/depth_test_cropped/" + str(ind))
		else:
			for i in range(len(image_ids)):
				ind = image_ids[i]
				imgs.append(path_to_dataset + "/rgb_train_cropped/" + str(ind))
				depths.append(path_to_dataset + "/depth_train_cropped/" + str(ind))

		self.image_ids = image_ids
		self.images = imgs
		self.depths = depths


	def __getitem__(self, i):

		# prepare image [H, W, 3]
		#img = np.transpose(self.images[i], axes=[2, 1, 0])
		# img = img.astype(np.uint8)

		# prepare depth map [H, W]
		# depth = np.transpose(self.depths[i], axes=[1, 0])
		# depth = (depth / depth.max()) * 255
		# depth = depth.astype(np.float32)
		# depth = depth / 4.0  # Normalize

		# extract masks of instances [H, W, N] and labels [N]

		image = np.load(self.images[i]+'.npy')
		depth = np.load(self.depths[i]+'.npy')/4

		kernel = np.ones((5, 5), np.uint8)
		edges = cv2.Canny(image, 80, 200)
		thick_edges = cv2.dilate(edges, kernel, iterations=1)
		thick_edges[np.where(thick_edges == 255)] = 1


		#if self.transform:
		#	image, depth = self.transform_test(image, depth)


		image, window, scale, padding = utils.resize_image(
			image,
			min_dim=self.config.IMAGE_MAX_DIM,
			max_dim=self.config.IMAGE_MAX_DIM,
			padding=self.config.IMAGE_PADDING)


		depth, window, scale, padding = utils.resize_depth(
			depth,
			min_dim=self.config.IMAGE_MAX_DIM,
			max_dim=self.config.IMAGE_MAX_DIM,
			padding=self.config.IMAGE_PADDING)

		thick_edges, window, scale, padding = utils.resize_depth(
			thick_edges,
			min_dim=self.config.IMAGE_MAX_DIM,
			max_dim=self.config.IMAGE_MAX_DIM,
			padding=self.config.IMAGE_PADDING)

		#if self.augment:

			# Horizontal flip
			#if np.random.randint(0, 1):
			#	image = np.fliplr(image)
			#	depth = np.fliplr(depth)
			#	pass
			#pass

		if self.augmentation:
			import imgaug

			# Augmenters that are safe to apply to masks
			# Some, such as Affine, have settings that make them unsafe, so always
			# test your augmentation on masks
			MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
							   "Fliplr", "Flipud", "CropAndPad",
							   "Affine", "PiecewiseAffine"]

			def hook(images, augmenter, parents, default):
				"""Determines which augmenters to apply to masks."""
				return augmenter.__class__.__name__ in MASK_AUGMENTERS

			# Store shapes before augmentation to compare
			image_shape = image.shape
			depth_shape = depth.shape
			# Make augmenters deterministic to apply similarly to images and masks
			det = self.augmentation.to_deterministic()
			image = det.augment_image(image)
			depth = det.augment_image(depth,
									 hooks=imgaug.HooksImages(activator=hook))
			# Verify that shapes didn't change
			assert image.shape == image_shape, "Augmentation shouldn't change image size"
			assert depth.shape == depth_shape, "Augmentation shouldn't change depth size"


		#depth = np.concatenate([np.zeros((80, 640)), depth, np.zeros((80, 640))], axis=0)




		#blurred_img = gaussian_filter(image, sigma=7)

		image = utils.mold_image(image.astype(np.float32), self.config)
		image = image.transpose((2, 0, 1)).astype(np.float32)
		depth = depth.astype(np.float32)
		thick_edges = thick_edges.astype(np.float32)


		#w = 448
		#image = image[:, :w, :]
		#depth = depth[:w, :]

		info = [image, thick_edges, depth]

		return info

	def __len__(self):
		return len(self.image_ids)


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


def load_image_gt(config, image_id, image, depth, mask, class_ids, augment=False,
				  use_mini_mask=False, augmentation=None):
	"""Load and return ground truth data for an image (image, mask, bounding boxes).

	augment: If true, apply random image augmentation. Currently, only
		horizontal flipping is offered.
	use_mini_mask: If False, returns full-size masks that are the same height
		and width as the original image. These can be big, for example
		1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
		224x224 and are generated by extracting the bounding box of the
		object and resizing it to MINI_MASK_SHAPE.

	Returns:
	image: [height, width, 3]
	shape: the original shape of the image before resizing and cropping.
	class_ids: [instance_count] Integer class IDs
	bbox: [instance_count, (y1, x1, y2, x2)]
	mask: [height, width, instance_count]. The height and width are those
		of the image unless use_mini_mask is True, in which case they are
		defined in MINI_MASK_SHAPE.
	"""
	## Load image and mask
	shape = image.shape
	image, window, scale, padding = utils.resize_image(
		image,
		min_dim=config.IMAGE_MAX_DIM,
		max_dim=config.IMAGE_MAX_DIM,
		padding=config.IMAGE_PADDING)

	mask = utils.resize_mask(mask, scale, padding)

	## Random horizontal flips.
	if augment:
		if np.random.randint(0, 1):
			image = np.fliplr(image)
			mask = np.fliplr(mask)
			depth = np.fliplr(depth)
			pass
		pass

	if augmentation:
		import imgaug

		# Augmenters that are safe to apply to masks
		# Some, such as Affine, have settings that make them unsafe, so always
		# test your augmentation on masks
		#MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
		#				   "Fliplr", "Flipud", "CropAndPad",
		#				   "Affine", "PiecewiseAffine"]

		#def hook(images, augmenter, parents, default):
		#	"""Determines which augmenters to apply to masks."""
		#	return augmenter.__class__.__name__ in MASK_AUGMENTERS

		# Store shapes before augmentation to compare
		image_shape = image.shape
		mask_shape = mask.shape
		depth_shape = depth.shape
		# Make augmenters deterministic to apply similarly to images and masks
		det = augmentation.to_deterministic()
		image = det.augment_image(image)
		#depth = det.augment_image(depth, hooks=imgaug.HooksImages(activator=hook))
		#mask = det.augment_image(mask.astype(np.uint8),
		#						 hooks=imgaug.HooksImages(activator=hook))
		depth = det.augment_image(depth)
		mask = det.augment_image(mask.astype(np.uint8))
		# Verify that shapes didn't change
		assert image.shape == image_shape, "Augmentation shouldn't change image size"
		assert depth.shape == depth_shape, "Augmentation shouldn't change depth size"
		assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
		# Change mask back to bool
		mask = mask.astype(np.bool)

	## Bounding boxes. Note that some boxes might be all zeros
	## if the corresponding mask got cropped out.
	## bbox: [num_instances, (y1, x1, y2, x2)]
	bbox = utils.extract_bboxes(mask)
	## Resize masks to smaller size to reduce memory usage
	if use_mini_mask:
		mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)
		pass

	active_class_ids = np.ones(config.NUM_CLASSES, dtype=np.int32)
	## Image meta data
	image_meta = utils.compose_image_meta(image_id, shape, window, active_class_ids)

	return image, image_meta, class_ids, bbox, mask, depth

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
	"""Given the anchors and GT boxes, compute overlaps and identify positive
	anchors and deltas to refine them to match their corresponding GT boxes.

	anchors: [num_anchors, (y1, x1, y2, x2)]
	gt_class_ids: [num_gt_boxes] Integer class IDs.
	gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

	Returns:
	rpn_match: [N] (int32) matches between anchors and GT boxes.
			   1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	"""
	## RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
	rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
	## RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
	rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

	## Handle COCO crowds
	## A crowd box in COCO is a bounding box around several instances. Exclude
	## them from training. A crowd box is given a negative class ID.
	no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

	## Compute overlaps [num_anchors, num_gt_boxes]
	overlaps = utils.compute_overlaps(anchors, gt_boxes)

	## Match anchors to GT Boxes
	## If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
	## If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
	## Neutral anchors are those that don't match the conditions above,
	## and they don't influence the loss function.
	## However, don't keep any GT box unmatched (rare, but happens). Instead,
	## match it to the closest anchor (even if its max IoU is < 0.3).
	#
	## 1. Set negative anchors first. They get overwritten below if a GT box is
	## matched to them. Skip boxes in crowd areas.
	anchor_iou_argmax = np.argmax(overlaps, axis=1)
	anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
	rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
	## 2. Set an anchor for each GT box (regardless of IoU value).
	## TODO: If multiple anchors have the same IoU match all of them
	gt_iou_argmax = np.argmax(overlaps, axis=0)
	rpn_match[gt_iou_argmax] = 1
	## 3. Set anchors with high overlap as positive.
	rpn_match[anchor_iou_max >= 0.7] = 1

	## Subsample to balance positive and negative anchors
	## Don't let positives be more than half the anchors
	ids = np.where(rpn_match == 1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
	if extra > 0:
		## Reset the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0
	## Same for negative proposals
	ids = np.where(rpn_match == -1)[0]
	extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
						np.sum(rpn_match == 1))
	if extra > 0:
		## Rest the extra ones to neutral
		ids = np.random.choice(ids, extra, replace=False)
		rpn_match[ids] = 0

	## For positive anchors, compute shift and scale needed to transform them
	## to match the corresponding GT boxes.
	ids = np.where(rpn_match == 1)[0]
	ix = 0  ## index into rpn_bbox
	## TODO: use box_refinment() rather than duplicating the code here
	for i, a in zip(ids, anchors[ids]):
		## Closest gt box (it might have IoU < 0.7)
		gt = gt_boxes[anchor_iou_argmax[i]]

		## Convert coordinates to center plus width/height.
		## GT Box
		gt_h = gt[2] - gt[0]
		gt_w = gt[3] - gt[1]
		gt_center_y = gt[0] + 0.5 * gt_h
		gt_center_x = gt[1] + 0.5 * gt_w
		## Anchor
		a_h = a[2] - a[0]
		a_w = a[3] - a[1]
		a_center_y = a[0] + 0.5 * a_h
		a_center_x = a[1] + 0.5 * a_w

		## Compute the bbox refinement that the RPN should predict.
		rpn_bbox[ix] = [
			(gt_center_y - a_center_y) / a_h,
			(gt_center_x - a_center_x) / a_w,
			np.log(gt_h / a_h),
			np.log(gt_w / a_w),
		]
		## Normalize
		rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
		ix += 1

	return rpn_match, rpn_bbox
