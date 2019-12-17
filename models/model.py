"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import datetime
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import logging

import utils
import visualize
from tensorboardX import SummaryWriter
from nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import cv2
from models.modules import *
from utils import *
from nyu import NYUDataset


############################################################
#  Logging Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
	prints it's shape, min, and max values.
	"""
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
        print(text)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total       - Required  : total iterations (Int)
		prefix      - Optional  : prefix string (Str)
		suffix      - Optional  : suffix string (Str)
		decimals    - Optional  : positive number of decimals in percent complete (Int)
		length      - Optional  : character length of bar (Int)
		fill        - Optional  : bar fill character (Str)
	"""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\n')
    # Print New Line on Complete
    if iteration == total:
        print()


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor[:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool), dim=0)
    return tensor[unique_bool.data]


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
	"""

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

	Returns:
		[N, (height, width)]. Where N is the number of stages
	"""
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])


############################################################
#  FPN Graph
############################################################

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels, bilinear_upsampling=False):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.bilinear_upsampling = bilinear_upsampling
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P6 = nn.MaxPool2d(kernel_size=1, stride=2)
        self.P5_conv1 = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P4_conv1 = nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P3_conv1 = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )
        self.P2_conv1 = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        #print(" FPN start: ", x.shape)
        x = self.C1(x)
        #print(" C1 first conv: ", x.shape)
        x = self.C2(x)
        #print(" C2 second conv: ", x.shape)
        c2_out = x
        x = self.C3(x)
        #print(" C3 third conv: ", x.shape)
        c3_out = x
        x = self.C4(x)
        #print(" C4 fourth conv: ", x.shape)
        c4_out = x
        x = self.C5(x)
        #print(" C5 fifty conv: ", x.shape)

        p5_out = self.P5_conv1(x)
        #print(" FPN5 second conv: ", x.shape)

        if self.bilinear_upsampling:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2, mode='bilinear')
            #print(" FPN4 second conv: ", p4_out.shape)
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2, mode='bilinear')
            #print(" FPN3 second conv: ", p3_out.shape)
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2, mode='bilinear')
            #print(" FPN2 second conv: ", p2_out.shape)
        else:
            p4_out = self.P4_conv1(c4_out) + F.upsample(p5_out, scale_factor=2)
            p3_out = self.P3_conv1(c3_out) + F.upsample(p4_out, scale_factor=2)
            p2_out = self.P2_conv1(c2_out) + F.upsample(p3_out, scale_factor=2)
            pass

        p5_out = self.P5_conv2(p5_out)
        #print(" FPN5 second conv2: ", p5_out.shape)
        p4_out = self.P4_conv2(p4_out)
        #print(" FPN4 second conv2: ", p4_out.shape)
        p3_out = self.P3_conv2(p3_out)
        #print(" FPN3 second conv2: ", p3_out.shape)
        p2_out = self.P2_conv2(p2_out)
        # print(" FPN2 second conv2: ", p2_out.shape)

        ## P6 is used for the 5th anchor scale in RPN. Generated by
        ## subsampling from P5 with stride of 2.
        p6_out = self.P6(p5_out)

        return [p2_out, p3_out, p4_out, p5_out, p6_out]


############################################################
#  Resnet Graph
############################################################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, architecture, stage5=False, numInputChannels=3):
        super(ResNet, self).__init__()
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck
        self.stage5 = stage5

        self.C1 = nn.Sequential(
            nn.Conv2d(numInputChannels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.C2 = self.make_layer(self.block, 64, self.layers[0])
        self.C3 = self.make_layer(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layer(self.block, 256, self.layers[2], stride=2)
        if self.stage5:
            self.C5 = self.make_layer(self.block, 512, self.layers[3], stride=2)
        else:
            self.C5 = None

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.C3(x)
        x = self.C4(x)
        x = self.C5(x)
        return x

    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]

    def make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion, eps=0.001, momentum=0.01),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


############################################################
#  Proposal Layer
############################################################

def original_apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
	boxes: [N, 4] where each row is y1, x1, y2, x2
	deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
	"""
    ## Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    ## Apply deltas
    center_y = center_y + deltas[:, 0] * height
    center_x = center_x + deltas[:, 1] * width
    height = height * torch.exp(deltas[:, 2])
    width = width * torch.exp(deltas[:, 3])
    ## Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)
    return result

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    :param boxes: [..., 4] where last dimension is y1, x1, y2, x2
    :param deltas: [..., 4] where last dimension is [dy, dx, log(dh), log(dw)]
    :return: boxes as a [..., 4] tensor
    """
    # Convert to y, x, h, w
    height = boxes[..., 2] - boxes[..., 0]
    width = boxes[..., 3] - boxes[..., 1]
    center_y = boxes[..., 0] + 0.5 * height
    center_x = boxes[..., 1] + 0.5 * width
    #print("height, weight, center: ", height, width, center_y, center_x)
    # print("Applying box deltas: ", "h, w: ", height.shape, width.shape, "boxes", boxes.shape, "deltas:", deltas.shape)
    # Apply deltas
    center_y = center_y + (deltas[..., 0] * height)
    center_x = center_x + (deltas[..., 1] * width)
    height = torch.exp(deltas[..., 2]) * height
    width = torch.exp(deltas[..., 3]) * width
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=-1)
    return result


def orig_clip_boxes(boxes, window):
    """
	boxes: [N, 4] each col is y1, x1, y2, x2
	window: [4] in the form y1, x1, y2, x2
	"""
    boxes = torch.stack( \
        [boxes[:, 0].clamp(float(window[0]), float(window[2])),
         boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])),
         boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
    return boxes

def clip_boxes(boxes, window):
    """
    Clip boxes to lie within window
    :param boxes: [N, 4] each col is y1, x1, y2, x2
    :param window: [4] in the form y1, x1, y2, x2
    :return: clipped boxes as a [N, 4] tensor
    """
    boxes = torch.stack( \
        [boxes[..., 0].clamp(float(window[0]), float(window[2])),
         boxes[..., 1].clamp(float(window[1]), float(window[3])),
         boxes[..., 2].clamp(float(window[0]), float(window[2])),
         boxes[..., 3].clamp(float(window[1]), float(window[3]))], dim=-1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
	to the second stage. Filtering is done based on anchor scores and
	non-max suppression to remove overlaps. It also applies bounding
	box refinement detals to anchors.

	Inputs:
		rpn_probs: [batch, anchors, (bg prob, fg prob)]
		rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

	Returns:
		Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
	"""

    ## Currently only supports batchsize 1

    #print("===> Proposal layer! ")
    #print("inputs before squeeze: ", inputs[0].shape, inputs[1].shape)

    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    #print("inputs after squeeze: ", inputs[0].shape, inputs[1].shape)

    ## Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    ## Box deltas [batch, num_rois, 4]
    deltas = inputs[1]

    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev
    ## Improve performance by trimming to top anchors by score
    ## and doing the rest on the smaller subset.
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.data, :]
    anchors = anchors[order.data, :]

    ## Apply deltas to anchors to get refined anchors.
    ## [batch, N, (y1, x1, y2, x2)]
    #print("anchors: ", anchors.shape, " deltas: ", deltas.shape)
    boxes = apply_box_deltas(anchors, deltas)

    ## Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    ## Filter out small boxes
    ## According to Xinlei Chen's paper, this reduces detection accuracy
    ## for small objects, so we're skipping it.

    ## Non-max suppression
    keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)

    keep = keep[:proposal_count]
    boxes = boxes[keep, :]

    ## Normalize dimensions to range of 0 to 1.
    norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    ## Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes

# batch proposal
def proposal_layer_batch(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
	to the second stage. Filtering is done based on anchor scores and
	non-max suppression to remove overlaps. It also applies bounding
	box refinement detals to anchors.

	Inputs:
		rpn_probs: [batch, anchors, (bg prob, fg prob)]
		rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]

	Returns:
		Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
	"""

    scores = inputs[0][:,:,1]

    ## Box deltas [batch, num_rois, 4]
    deltas = inputs[1]
    n_samples = deltas.shape[0]

    # print("scores: ", scores.shape, " deltas: ", deltas.shape)

    std_dev = torch.tensor(np.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4]), requires_grad=False, dtype=torch.float)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()


    deltas = deltas * std_dev
    ## Improve performance by trimming to top anchors by score
    ## and doing the rest on the smaller subset.
    # print(" anchors size: ", anchors.size())
    pre_nms_limit = min(6000, anchors.size()[0])
    scores, order = scores.sort(dim=1, descending=True)
    order = order[:, :pre_nms_limit]
    scores = scores[:, :pre_nms_limit]
    #deltas = deltas[order.data, :]
    # print("deltas: ", deltas.shape, "order: ", order.shape, "n_samples: ", n_samples)
    # print("order type: ", order.dtype)
    deltas = deltas[torch.arange(n_samples, dtype=torch.long)[:, None], order, :]
    anchors = anchors[order, :]

    # print("scores: ", scores.shape, " deltas: ", deltas.shape, " std dev: ", std_dev.shape)

    ## Apply deltas to anchors to get refined anchors.
    ## [batch, N, (y1, x1, y2, x2)]
    # print("anchors: ", anchors.shape, " deltas: ", deltas.shape)
    boxes = apply_box_deltas(anchors, deltas)

    ## Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
    height, width = config.IMAGE_SHAPE[:2]
    window = np.array([0, 0, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    ## Filter out small boxes
    ## According to Xinlei Chen's paper, this reduces detection accuracy
    ## for small objects, so we're skipping it.

    ## Non-max suppression
    retained = []
    rois_per_sample = []
    max_n_rois = 0
    for sample_i in range(n_samples):
        keep = nms(torch.cat((boxes[sample_i], scores[sample_i].unsqueeze(1)), 1).detach(), nms_threshold)
        keep = keep[:proposal_count]
        n_rois = keep.shape[0]
        max_n_rois = max(max_n_rois, n_rois)
        retained.append(keep)
        rois_per_sample.append(n_rois)

    # Normalize dimensions to range of 0 to 1.
    norm = torch.tensor(np.array([height, width, height, width]), dtype=torch.float)
    retained_norm_boxes = torch.zeros(n_samples, max_n_rois, 4, dtype=torch.float)
    retained_scores = torch.zeros(n_samples, max_n_rois, dtype=torch.float)
    if config.GPU_COUNT:
        norm = norm.cuda()
        retained_norm_boxes = retained_norm_boxes.cuda()
        retained_scores = retained_scores.cuda()

    for sample_i, (n_rois, keep) in enumerate(zip(rois_per_sample, retained)):
        retained_norm_boxes[sample_i, :n_rois, :] = boxes[sample_i, keep, :] / norm
        retained_scores[sample_i, :n_rois] = scores[sample_i, keep]

    # print("retained boxes: ", retained_norm_boxes.shape)

    return retained_norm_boxes, retained_scores, rois_per_sample

############################################################
#  ROIAlign Layer
############################################################

def orig_pyramid_roi_align(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

	Params:
	- pool_size: [height, width] of the output pooled regions. Usually [7, 7]
	- image_shape: [height, width, channels]. Shape of input image in pixels

	Inputs:
	- boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
			 coordinates.
	- Feature maps: List of feature maps from different levels of the pyramid.
					Each is [batch, channels, height, width]

	Output:
	Pooled regions in the shape: [num_boxes, height, width, channels].
	The width and height are those specific in the pool_shape in the layer
	constructor.
	"""

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # print(" ---> pyramid roi align :")
    # print( " h, w : ", h.shape, w.shape )

    ## Equation 1 in the Feature Pyramid Networks paper. Account for
    ## the fact that our coordinates are normalized here.
    ## e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    # print(" image area: ", image_area)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    # print(" roilevel 1: ", roi_level.shape, " ex: ", roi_level[0])
    roi_level = roi_level.round().int()
    # print(" roilevel 2: ", roi_level.shape, " ex: ", roi_level[0])
    roi_level = roi_level.clamp(2, 5)
    # print(" roilevel 3: ", roi_level.shape, " ex: ", roi_level[0])

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix.data, :]

        ## Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        ## Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        ## Crop and Resize
        ## From Mask R-CNN paper: "We sample four regular locations, so
        ## that we can evaluate either max or average pooling. In fact,
        ## interpolating only a single value at each bin center (without
        ## pooling) is nearly as effective."
        #
        ## Here we use the simplified approach of a single value per bin,
        ## which is how it's done in tf.crop_and_resize()
        ## Result: [batch * num_boxes, pool_height, pool_width, channels]
        ind = Variable(torch.zeros(level_boxes.size()[0]), requires_grad=False).int()
        if level_boxes.is_cuda:
            ind = ind.cuda()
        # print(" pooling layer: ", i, " box size: ", ind.shape, " feature map: ", feature_maps[i].shape)
        feature_maps[i] = feature_maps[i].unsqueeze(0)  # CropAndResizeFunction needs batch dimension
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, ind)
        # print(" cropped feature: ", pooled_features.shape)
        pooled.append(pooled_features)

    # print(" pooled: ", pooled[0].shape, pooled[1].shape, pooled[2].shape)
    ## Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)
    ## Pack box_to_level mapping into one array and add another
    ## column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    ## Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled

def flatten_detections_with_sample_indices(n_dets_per_sample, *dets):
    """
    Flatten detection tensors from shape [batch, detections, ...] with zero padding
    in the detections axis for unused detections. The number of used detections in
    each sample is specified in n_dets_per_sample.
    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param dets: tensors of detections, where each tensor is of shape [batch, detection, ...]
    :return: (flat_dets0, flat_dets1, ...flat_detsN, sample_indices) where
        flat_dets0..flat_detsN are tensors of shape [batch/detection, ...]
        sample_indices gives the index of the sample to which each detection in the preceeding
        arrays came from
    """
    sample_indices = []
    for sample_i, n_dets in enumerate(n_dets_per_sample):
        assign = torch.ones(n_dets).int() * sample_i
        sample_indices.append(assign)
    sample_indices = torch.cat(sample_indices, dim=0).to(dets[0].device)

    # Flatten detections
    flat_dets = []
    for det in dets:
        flat = []
        for sample_i, n_dets in enumerate(n_dets_per_sample):
            if det.size() and n_dets > 0:
                flat.append(det[sample_i, :n_dets, ...])
        if len(flat) > 0:
            flat_dets.append(torch.cat(flat, dim=0))
        else:
            empty = det.new(torch.Size()).zero_()
            flat_dets.append(empty)

    return tuple(flat_dets) + (sample_indices,)

def flatten_detections(n_dets_per_sample, *dets):
    """
    Flatten detection tensors from shape [batch, detections, ...] with zero padding
    in the detections axis for unused detections. The number of used detections in
    each sample is specified in n_dets_per_sample.
    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param dets: tensors of detections, where each tensor is of shape [batch, detection, ...]
    :return: (flat_dets0, flat_dets1, ...flat_detsN) where
        flat_dets0..flat_detsN are tensors of shape [batch/detection, ...]
    """
    # Flatten detections
    flat_dets = []
    for det in dets:
        flat = []
        #print("det shape: ", det.shape)
        for sample_i, n_dets in enumerate(n_dets_per_sample):
            if det.size() and n_dets > 0:
                flat.append(det[sample_i, :n_dets, ...])
        if len(flat) > 0:
            flat_dets.append(torch.cat(flat, dim=0))
        else:
            empty = det.new(torch.Size()).zero_()
            flat_dets.append(empty)

    return tuple(flat_dets)

def unflatten_detections(n_dets_per_sample, *flat_dets):
    """
    Un-flatten the detections in the tensors in flat_dets.
    :param n_dets_per_sample: Number of detections in each sample in the batch
    :param flat_dets: tensors of detections, where each tensor is of shape [sample/detection, ...]
    :return: dets where
        dets is a list of tensors of shape [batch, detection, ...]
        with zero-padding for unused detections
    """

    if len(n_dets_per_sample) == 1:
        return [d[None] for d in flat_dets]
    else:
        max_n_dets = max(n_dets_per_sample)
        n_samples = len(n_dets_per_sample)

        dets = [fdet.new(torch.Size((n_samples, max_n_dets, *fdet.size()[1:]))).zero_().to(fdet.device)
                for fdet in flat_dets]

        offset = 0
        for sample_i in range(n_samples):
            n_dets = n_dets_per_sample[sample_i]
            if n_dets > 0:
                for det, fdet in zip(dets, flat_dets):
                    det[sample_i, :n_dets] = fdet[offset:offset + n_dets]
                offset += n_dets
        return dets

def concatenate_detection_tensors(dets):
    """
    Concatenate detection tensors along the batch axis
    Each tensor is a Torch tensor of shape [batch, detection, ...]
    Concatenates torch tensors along the batch axis, zero-padding the detection axis if necessary.
    :param dets: list of Torch tensors
    :return: (detections, n_dets_per_sample) where
        detections is the concatenated tensor
        n_dets_per_sample is a list giving the number of valid detections per batch sample
    """
    n_dets_per_sample = [(d.size()[1] if len(d.size())>=2 else 0) for d in dets]

    if len(dets) == 1:
        return dets[0], n_dets_per_sample

    #print(" n dets per sample ", n_dets_per_sample)
    #print("dets: ", len(dets))
    #print(dets[0].shape, dets[1].shape)

    max_dets = max(n_dets_per_sample)
    #print("max dets: ", max_dets)
    det_shape = ()
    example_det = dets[0]
    for d, n_dets in zip(dets, n_dets_per_sample):
        if n_dets != 0:
            det_shape = d.size()[2:]
            example_det = d
            break
    if max_dets > 0:
        padded_dets = []
        for det, n_dets in zip(dets, n_dets_per_sample):
            if n_dets < max_dets:
                z = example_det.new(1, max_dets - n_dets, *det_shape).zero_().to(det.device)
                if n_dets == 0:
                    padded_dets.append(z)
                else:
                    padded_dets.append(torch.cat([det, z], dim=1))
            else:
                padded_dets.append(det)
        return torch.cat(padded_dets, dim=0), n_dets_per_sample
    else:
        return example_det.new(0), n_dets_per_sample

def concatenate_detections(*dets):
    """
    Concatenate detections along the batch axis
    Each entry in det_tuples is a tuple of detections for the corresponding sample
    :param dets: each item is a list of Torch tensors
    :return: (detections, n_dets_per_sample), where:
        detections is a tuple of concatenated tensors
        n_dets_per_sample is a list giving the number of valid detections per batch sample
    """
    detections = []
    n_dets_per_sample = []
    for dets_by_sample in dets:
        cat_dets, n_dets_per_sample = concatenate_detection_tensors(list(dets_by_sample))
        detections.append(cat_dets)
    return tuple(detections), n_dets_per_sample

_EMPTY_SIZES = {torch.Size([]), torch.Size([0])}

def not_empty(tensor):
    return tensor.size() not in _EMPTY_SIZES

def is_empty(tensor):
    return tensor.size() in _EMPTY_SIZES

def pyramid_roi_align(inputs, pool_size, image_shape, n_boxes_per_sample):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

	Params:
	- pool_size: [height, width] of the output pooled regions. Usually [7, 7]
	- image_shape: [height, width, channels]. Shape of input image in pixels

	Inputs:
	- boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
			 coordinates.
	- Feature maps: List of feature maps from different levels of the pyramid.
					Each is [batch, channels, height, width]

	Output:
	Pooled regions in the shape: [num_boxes, height, width, channels].
	The width and height are those specific in the pool_shape in the layer
	constructor.
	"""

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    feature_maps = inputs[1:]

    #print(" boxes: ", boxes.shape)
    #print("n boxes: ", n_boxes_per_sample)
    boxes_flat, box_sample_indices = flatten_detections_with_sample_indices(n_boxes_per_sample, boxes)

    #print("flat box: ", boxes_flat.shape)

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes_flat.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    # print(" ---> pyramid roi align :")
    # print( " h, w : ", h.shape, w.shape )

    ## Equation 1 in the Feature Pyramid Networks paper. Account for
    ## the fact that our coordinates are normalized here.
    ## e.g. a 224x224 ROI (in pixels) maps to P4
    image_area = Variable(torch.FloatTensor([float(image_shape[0] * image_shape[1])]), requires_grad=False)
    # print(" image area: ", image_area)
    if boxes.is_cuda:
        image_area = image_area.cuda()
    roi_level = 4 + log2(torch.sqrt(h * w) / (224.0 / torch.sqrt(image_area)))
    # print(" roilevel 1: ", roi_level.shape, " ex: ", roi_level[0])
    roi_level = roi_level.round().int()
    # print(" roilevel 2: ", roi_level.shape, " ex: ", roi_level[0])
    roi_level = roi_level.clamp(2, 5)
    # print(" roilevel 3: ", roi_level.shape, " ex: ", roi_level[0])

    ## Loop through levels and apply ROI pooling to each. P2 to P5.

    pooled = []
    box_to_level = []
    for i, level in enumerate(range(2, 6)):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes_flat[ix.data, :]
        level_sample_indices = box_sample_indices[ix]

        ## Keep track of which box is mapped to which level
        box_to_level.append(ix.data)

        ## Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()

        #print("feats: ", feature_maps[i].shape)
        #print("level boxes: ", level_boxes.shape)
        #print("level_sample_indices: ", level_sample_indices.shape)
        ## Crop and Resize
        ## From Mask R-CNN paper: "We sample four regular locations, so
        ## that we can evaluate either max or average pooling. In fact,
        ## interpolating only a single value at each bin center (without
        ## pooling) is nearly as effective."
        #
        ## Here we use the simplified approach of a single value per bin,
        ## which is how it's done in tf.crop_and_resize()
        ## Result: [batch * num_boxes, pool_height, pool_width, channels]

        # print(" pooling layer: ", i, " box size: ", ind.shape, " feature map: ", feature_maps[i].shape)
        pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(feature_maps[i], level_boxes, level_sample_indices)
        #print(" cropped feature: ", pooled_features.shape)
        pooled.append(pooled_features)

    # print(" pooled: ", pooled[0].shape, pooled[1].shape, pooled[2].shape)
    ## Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)
    ## Pack box_to_level mapping into one array and add another
    ## column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    ## Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled


def coordinates_roi(inputs, pool_size, image_shape):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

	Params:
	- pool_size: [height, width] of the output pooled regions. Usually [7, 7]
	- image_shape: [height, width, channels]. Shape of input image in pixels

	Inputs:
	- boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
			 coordinates.
	- Feature maps: List of feature maps from different levels of the pyramid.
					Each is [batch, channels, height, width]

	Output:
	Pooled regions in the shape: [num_boxes, height, width, channels].
	The width and height are those specific in the pool_shape in the layer
	constructor.
	"""

    ## Currently only supports batchsize 1
    for i in range(len(inputs)):
        inputs[i] = inputs[i].squeeze(0)

    ## Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
    boxes = inputs[0]

    ## Feature Maps. List of feature maps from different level of the
    ## feature pyramid. Each is [batch, height, width, channels]
    cooridnates = inputs[1]

    ## Assign each ROI to a level in the pyramid based on the ROI area.
    y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    h = y2 - y1
    w = x2 - x1

    ## Loop through levels and apply ROI pooling to each. P2 to P5.
    pooled = []
    ## Stop gradient propogation to ROI proposals
    boxes = boxes.detach()

    ind = Variable(torch.zeros(boxes.size()[0]), requires_grad=False).int()
    if boxes.is_cuda:
        ind = ind.cuda()
    cooridnates = cooridnates.unsqueeze(0)  ## CropAndResizeFunction needs batch dimension
    pooled_features = CropAndResizeFunction(pool_size, pool_size, 0)(cooridnates, boxes, ind)

    return pooled_features


############################################################
##  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	"""
    ## 1. Tile boxes2 and repeate boxes1. This allows us to compare
    ## every boxes1 against every boxes2 without loops.
    ## TF doesn't have an equivalent to np.repeate() so simulate it
    ## using tf.tile() and tf.reshape.
    boxes1_repeat = boxes2.size()[0]
    boxes2_repeat = boxes1.size()[0]
    boxes1 = boxes1.repeat(1, boxes1_repeat).view(-1, 4)
    boxes2 = boxes2.repeat(boxes2_repeat, 1)

    ## 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
    y1 = torch.max(b1_y1, b2_y1)[:, 0]
    x1 = torch.max(b1_x1, b2_x1)[:, 0]
    y2 = torch.min(b1_y2, b2_y2)[:, 0]
    x2 = torch.min(b1_x2, b2_x2)[:, 0]
    zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
    if y1.is_cuda:
        zeros = zeros.cuda()
    intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

    ## 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area[:, 0] + b2_area[:, 0] - intersection

    ## 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = iou.view(boxes2_repeat, boxes1_repeat)

    return overlaps


def detection_target_layer_(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
	and masks for each.

	Inputs:
	proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
			   be zero padded if there are not enough proposals.
	gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
	gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
			  coordinates.
	gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

	Returns: Target ROIs and corresponding class IDs, bounding box shifts,
	and masks.
	rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
		  coordinates
	target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
	target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
					(dy, dx, log(dh), log(dw), class_id)]
				   Class-specific bbox refinments.
	target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
				 Masks cropped to bbox boundaries and resized to neural
				 network output size.
	"""

    ## Currently only supports batchsize 1
    proposals = proposals.squeeze(0)
    gt_class_ids = gt_class_ids.squeeze(0)
    gt_boxes = gt_boxes.squeeze(0)
    gt_masks = gt_masks.squeeze(0)
    no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
    if config.GPU_COUNT:
        no_crowd_bool = no_crowd_bool.cuda()

    ## Compute overlaps matrix [proposals, gt_boxes]
    overlaps = bbox_overlaps(proposals, gt_boxes)

    ## Determine postive and negative ROIs
    roi_iou_max = torch.max(overlaps, dim=1)[0]

    ## 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = roi_iou_max >= 0.5
    # print('positive count', positive_roi_bool.sum())

    ## Subsample ROIs. Aim for 33% positive
    ## Positive ROIs
    if positive_roi_bool.sum() > 0:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE *
                             config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data, :]

        ## Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data, :]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data, :]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        ## Compute bbox refinement for positive ROIs
        deltas = Variable(utils.box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT:
            std_dev = std_dev.cuda()
        deltas /= std_dev

        ## Assign positive ROIs to GT masks
        roi_masks = gt_masks[roi_gt_box_assignment.data]

        ## Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            ## Transform ROI corrdinates from normalized image space
            ## to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT:
            box_ids = box_ids.cuda()

        if config.NUM_PARAMETER_CHANNELS > 0:
            masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(
                roi_masks[:, :, :, 0].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = torch.round(masks)
            parameters = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(
                roi_masks[:, :, :, 1].contiguous().unsqueeze(1), boxes, box_ids).data, requires_grad=False).squeeze(1)
            masks = torch.stack([masks, parameters], dim=-1)
        else:
            masks = Variable(
                CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes,
                                                                                     box_ids).data,
                requires_grad=False).squeeze(1)
            masks = torch.round(masks)
            pass

    ## Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
    ## binary cross entropy loss.
    else:
        positive_count = 0

    ## 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool
    ## Negative ROIs. Add enough to maintain positive:negative ratio.
    if (negative_roi_bool > 0).sum() > 0 and positive_count > 0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT:
            rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else:
        negative_count = 0

    # print('count', positive_count, negative_count)
    # print(roi_gt_class_ids)

    ## Append negative ROIs and pad bbox deltas and masks that
    ## are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        if config.NUM_PARAMETER_CHANNELS > 0:
            zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1], 2),
                             requires_grad=False)
        else:
            zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                             requires_grad=False)
            pass
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
    elif positive_count > 0:
        rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count, 4), requires_grad=False).int()
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
        masks = zeros

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
            pass

    return rois, roi_gt_class_ids, deltas, masks

############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
		window: (y1, x1, y2, x2). The window in the image we want to clip to.
		boxes: [N, (y1, x1, y2, x2)]
	"""
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[2])), boxes[:, 1].clamp(float(window[1]), float(window[3])),
         boxes[:, 2].clamp(float(window[0]), float(window[2])), boxes[:, 3].clamp(float(window[1]), float(window[3]))],
        dim=-1)
    return boxes


def refine_detections_(rois, probs, deltas, window, config, return_indices=False, use_nms=1, one_hot=True):
    """Refine classified proposals and filter overlaps and return final
	detections.

	Inputs:
		rois: [N, (y1, x1, y2, x2)] in normalized coordinates
		probs: [N, num_classes]. Class probabilities.
		deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
				bounding box deltas.
		window: (y1, x1, y2, x2) in image coordinates. The part of the image
			that contains the image excluding the padding.

	Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
	"""

    ## Class IDs per ROI

    if len(probs.shape) == 1:
        class_ids = probs.long()
    else:
        _, class_ids = torch.max(probs, dim=1)
        pass

    ## Class probability of the top class of each ROI
    ## Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT:
        idx = idx.cuda()

    if len(probs.shape) == 1:
        class_scores = torch.ones(class_ids.shape)
        deltas_specific = deltas
        if config.GPU_COUNT:
            class_scores = class_scores.cuda()
    else:
        class_scores = probs[idx, class_ids.data]
        deltas_specific = deltas[idx, class_ids.data]
    ## Apply bounding box deltas
    ## Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()

    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)
    ## Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT:
        scale = scale.cuda()
    refined_rois = refined_rois * scale
    ## Clip boxes to image window
    refined_rois = clip_to_window(window, refined_rois)

    ## Round and cast to int since we're deadling with pixels now
    refined_rois = torch.round(refined_rois)

    ## TODO: Filter out boxes with zero area

    ## Filter out background boxes
    keep_bool = class_ids > 0

    ## Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE and False:
        keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)

    keep_bool = keep_bool & (refined_rois[:, 2] > refined_rois[:, 0]) & (refined_rois[:, 3] > refined_rois[:, 1])

    if keep_bool.sum() == 0:
        if return_indices:
            return torch.zeros((0, 10)).cuda(), torch.zeros(0).long().cuda(), torch.zeros((0, 4)).cuda()
        else:
            return torch.zeros((0, 10)).cuda()
        pass

    keep = torch.nonzero(keep_bool)[:, 0]

    if use_nms == 2:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        ixs = torch.arange(len(pre_nms_class_ids)).long().cuda()
        ## Sort
        ix_rois = pre_nms_rois
        ix_scores = pre_nms_scores
        ix_scores, order = ix_scores.sort(descending=True)
        ix_rois = ix_rois[order.data, :]

        nms_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
        nms_keep = keep[ixs[order[nms_keep].data].data]
        keep = intersect1d(keep, nms_keep)
    elif use_nms == 1:
        ## Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]

        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            ## Pick detections of this class
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:, 0]

            ## Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data, :]

            class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)

            ## Map indicies
            class_keep = keep[ixs[order[class_keep].data].data]

            if i == 0:
                nms_keep = class_keep
            else:
                nms_keep = unique1d(torch.cat((nms_keep, class_keep)))
        keep = intersect1d(keep, nms_keep)
    else:
        pass

    ## Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
    keep = keep[top_ids.data]
    # print('num detectinos', len(keep))

    ## Arrange output as [N, (y1, x1, y2, x2, class_id, score, parameters)]
    ## Coordinates are in image domain.
    result = torch.cat((refined_rois[keep.data],
                        class_ids[keep.data].unsqueeze(1).float(),
                        class_scores[keep.data].unsqueeze(1)), dim=1)

    if return_indices:
        ori_rois = rois * scale
        ori_rois = clip_to_window(window, ori_rois)
        ori_rois = torch.round(ori_rois)
        ori_rois = ori_rois[keep.data]
        return result, keep.data, ori_rois

    return result
    #return refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(), class_scores[keep.data].unsqueeze(1)


def detection_layer_(config, rois, mrcnn_class, mrcnn_bbox, image_meta, return_indices=False, use_nms=1,
                     one_hot=True):
    """Takes classified proposal boxes and their bounding box deltas and
	returns the final detection boxes.

	Returns:
	[batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
	"""

    ## Currently only supports batchsize 1
    rois = rois.squeeze(0)

    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    if len(mrcnn_class) == 0:
        if return_indices:
            return torch.zeros(0), torch.zeros(0), torch.zeros(0)
        else:
            return torch.zeros(0)

    return refine_detections_(rois, mrcnn_class[0], mrcnn_bbox[0], window, config,
                              return_indices=return_indices, use_nms=use_nms, one_hot=one_hot)

# batch version of detection layer for inference
def batch_detection_layer_(config, rois, mrcnn_class, mrcnn_bbox, image_metas, return_indices=False, use_nms=1, one_hot=True):

    device = mrcnn_bbox.device

    detections = []
    n_detections_total = 0
    #n_rois = config.DETECTION_MAX_INSTANCES
    for sample_i in range(config.BATCH_SIZE):
        _, _, window, _ = parse_image_meta(image_metas[sample_i].unsqueeze(0))
        detection = refine_detections_(rois, mrcnn_class[sample_i],
                                           mrcnn_bbox[sample_i],
                                           window,
                                           config,
                                           return_indices=return_indices,
                                           use_nms=use_nms, one_hot=one_hot)
        if detection is None:
            detections = torch.zeros([0], dtype=torch.float, device=device)
        else:
            n_detections_total += detections.size()[0]
            detections = detections.unsqueeze(0)
            detections.append(detections)

    if n_detections_total > 0:
        detections = torch.cat(detections, dim=0)
    else:
        detections = torch.zeros([0], dtype=torch.float, device=device)

    return detections


############################################################
#  Region Proposal Network
############################################################

class RPN(nn.Module):
    """Builds the model of Region Proposal Network.

	anchors_per_location: number of anchors per pixel in the feature map
	anchor_stride: Controls the density of anchors. Typically 1 (anchors for
				   every pixel in the feature map), or 2 (every other pixel).

	Returns:
		rpn_logits: [batch, H, W, 2] Anchor classifier logits (before softmax)
		rpn_probs: [batch, W, W, 2] Anchor classifier probabilities.
		rpn_bbox: [batch, H, W, (dy, dx, log(dh), log(dw))] Deltas to be
				  applied to anchors.
	"""

    def __init__(self, anchors_per_location, anchor_stride, depth):
        super(RPN, self).__init__()
        self.anchors_per_location = anchors_per_location
        self.anchor_stride = anchor_stride
        self.depth = depth

        self.padding = SamePad2d(kernel_size=3, stride=self.anchor_stride)
        self.conv_shared = nn.Conv2d(self.depth, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv_class = nn.Conv2d(512, 2 * anchors_per_location, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        self.conv_bbox = nn.Conv2d(512, 4 * anchors_per_location, kernel_size=1, stride=1)

    def forward(self, x):
        ## Shared convolutional base of the RPN
        x = self.relu(self.conv_shared(self.padding(x)))

        ## Anchor Score. [batch, anchors per location * 2, height, width].
        rpn_class_logits = self.conv_class(x)

        ## Reshape to [batch, 2, anchors]
        rpn_class_logits = rpn_class_logits.permute(0, 2, 3, 1)
        rpn_class_logits = rpn_class_logits.contiguous()
        rpn_class_logits = rpn_class_logits.view(x.size()[0], -1, 2)

        ## Softmax on last dimension of BG/FG.
        rpn_probs = self.softmax(rpn_class_logits)

        ## Bounding box refinement. [batch, H, W, anchors per location, depth]
        ## where depth is [x, y, log(w), log(h)]
        rpn_bbox = self.conv_bbox(x)

        ## Reshape to [batch, 4, anchors]
        rpn_bbox = rpn_bbox.permute(0, 2, 3, 1)
        rpn_bbox = rpn_bbox.contiguous()
        rpn_bbox = rpn_bbox.view(x.size()[0], -1, 4)

        return [rpn_class_logits, rpn_probs, rpn_bbox]


############################################################
#  Feature Pyramid Network Heads
############################################################

class Classifier(nn.Module):
    def __init__(self, depth, pool_size, image_shape, num_classes, debug=False):
        super(Classifier, self).__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        # self.conv1 = nn.Conv2d(self.depth + 64, 1024, kernel_size=self.pool_size, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, x, rois, n_rois_per_sample):
        # print(" ==== WELCOME TO CLASSIFIER ==== ")
        # print("rois, x: ", len(x), rois.shape)
        x = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape, n_rois_per_sample)
        # print(" after roi align: ", x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print(" end of convolutions: ", x.shape)

        x = x.view(-1, 1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)


        # print("bbox shape: ", mrcnn_bbox.shape, " probs: ", mrcnn_probs.shape)
        (mrcnn_class_logits,) = unflatten_detections(n_rois_per_sample, mrcnn_class_logits)
        (mrcnn_probs,) = unflatten_detections(n_rois_per_sample, mrcnn_probs)
        (mrcnn_bbox,) = unflatten_detections(n_rois_per_sample, mrcnn_bbox)

        # print(" ==== GOOD BYE! ==== ")
        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class Mask(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(Mask, self).__init__()
        self.config = config
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.padding = SamePad2d(kernel_size=3, stride=1)
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, n_rois_per_sample, pool_features=True):
        if pool_features:
            roi_features = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape, n_rois_per_sample)
        else:
            roi_features = x
            pass
        x = self.conv1(self.padding(roi_features))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(self.padding(x))
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(self.padding(x))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(self.padding(x))
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        #print(" this is mask: ", x.shape, n_rois_per_sample)
        (x,) = unflatten_detections(n_rois_per_sample, x)

        return x, roi_features


class Depth(nn.Module):

    # Changed version, no skip
    def modified_init__(self, num_output_channels=1):
        super(Depth, self).__init__()
        self.num_output_channels = num_output_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.depth_pred = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

        self.crop = True
        return

    # original init with skip connections for pyramid
    def __init__(self, num_output_channels=1):
        super(Depth, self).__init__()
        self.num_output_channels = num_output_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.depth_pred = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

        self.crop = True
        return

    # Remove the skip connections from pyramid features
    def modified_forward(self, feature_maps):
        if self.crop:
            padding = 5
            for c in range(2, 5):
                feature_maps[c] = feature_maps[c][:, :, padding * pow(2, c - 2):-padding * pow(2, c - 2)]
                continue
            pass
        ## print("FMap 0: ", feature_maps[0].shape)
        x = self.conv1(feature_maps[0])
        ## print("first conv of feature map: ", x.shape)
        x = self.deconv1(x)
        ## print("deconv conv: ", x.shape)
        #y = self.conv2(feature_maps[1])
        #print("conv of fmap 1:", y.shape)
        #x = torch.cat([y, x], dim=1)
        #print("concatenated: ", x.shape)
        x = self.deconv2(x)
        ## print("Deconv again: ", x.shape)
        if self.crop:
            x = x[:, :, 5:35]
        #y = self.conv3(feature_maps[2])
        #print("conv3 of map 2: ", y.shape)
        #print("vector together: ", (y, x).shape)
        #x = torch.cat([y, x], dim=1)
        #print("concated: ", x.shape)
        x = self.deconv3(x)
        ## print("deconv3: ", x.shape)
        x = self.deconv4(x)
        x = self.deconv5(x)
        #x = self.deconv4(torch.cat([self.conv4(feature_maps[3]), x], dim=1))
        #x = self.deconv5(torch.cat([self.conv5(feature_maps[4]), x], dim=1))
        ## print(" Normal before last conv ", x.shape)
        x = self.depth_pred(x)
        ## print(" Normal after last conv ", x.shape)

        if self.crop:
            x = torch.nn.functional.interpolate(x, size=(480, 640), mode='bilinear')
            zeros = torch.zeros((len(x), self.num_output_channels, 80, 640)).cuda()
            x = torch.cat([zeros, x, zeros], dim=2)
        else:
            x = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
            pass

        ## print(" Normal after interpolate ", x.shape, " crop? ", self.crop)

        return x

    def forward(self, feature_maps):
        if self.crop:
            padding = 5
            for c in range(2, 5):
                feature_maps[c] = feature_maps[c][:, :, padding * pow(2, c - 2):-padding * pow(2, c - 2)]
                continue
            pass
        # print("FMap 0: ", feature_maps[0].shape)
        x = self.conv1(feature_maps[0])
        # print("first conv of feature map: ", x.shape)
        x = self.deconv1(x)
        # print("deconv conv: ", x.shape)
        y = self.conv2(feature_maps[1])
        # print("conv of fmap 1:", y.shape)
        x = torch.cat([y, x], dim=1)
        # print("concatenated: ", x.shape)
        x = self.deconv2(x)
        # print("Deconv again: ", x.shape)
        if self.crop:
            x = x[:, :, 5:35]
        y = self.conv3(feature_maps[2])
        # print("conv3 of map 2: ", y.shape)
        # print("vector together: ", (y, x).shape)
        x = torch.cat([y, x], dim=1)
        # print("concated: ", x.shape)
        x = self.deconv3(x)
        # print("deconv3: ", x.shape)
        x = self.deconv4(torch.cat([self.conv4(feature_maps[3]), x], dim=1))
        x = self.deconv5(torch.cat([self.conv5(feature_maps[4]), x], dim=1))
        # print(" Normal before last conv ", x.shape)
        x = self.depth_pred(x)
        # print(" Normal after last conv ", x.shape)

        if self.crop:
            x = torch.nn.functional.interpolate(x, size=(480, 640), mode='bilinear')
            zeros = torch.zeros((len(x), self.num_output_channels, 80, 640)).cuda()
            x = torch.cat([zeros, x, zeros], dim=2)
        else:
            x = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
            pass

        # print(" Normal after interpolate ", x.shape, " crop? ", self.crop)

        return x



class Plane(nn.Module):
    def __init__(self, num_output_channels=3):
        super(Depth, self).__init__()
        self.num_output_channels = num_output_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True)
        )

        self.plane_pred = nn.Conv2d(64, num_output_channels, kernel_size=3, stride=1, padding=1)

        self.crop = True
        return

    def forward(self, feature_maps):
        if self.crop:
            padding = 5
            for c in range(2, 5):
                feature_maps[c] = feature_maps[c][:, :, padding * pow(2, c - 2):-padding * pow(2, c - 2)]
                continue
            pass
        # print("FMap 0: ", feature_maps[0].shape)
        x = self.conv1(feature_maps[0])
        # print("first conv of feature map: ", x.shape)
        x = self.deconv1(x)
        # print("deconv conv: ", x.shape)
        y = self.conv2(feature_maps[1])
        # print("conv of fmap 1:", y.shape)
        x = torch.cat([y, x], dim=1)
        # print("concatenated: ", x.shape)
        x = self.deconv2(x)
        # print("Deconv again: ", x.shape)
        if self.crop:
            x = x[:, :, 5:35]
        y = self.conv3(feature_maps[2])
        # print("conv3 of map 2: ", y.shape)
        # print("vector together: ", (y, x).shape)
        x = torch.cat([y, x], dim=1)
        # print("concated: ", x.shape)
        x = self.deconv3(x)
        # print("deconv3: ", x.shape)
        x = self.deconv4(torch.cat([self.conv4(feature_maps[3]), x], dim=1))
        x = self.deconv5(torch.cat([self.conv5(feature_maps[4]), x], dim=1))
        #print(" Normal before last conv ", x.shape)
        x = self.plane_pred(x)
        #print(" Normal after last conv ", x.shape)

        if self.crop:
            x = torch.nn.functional.interpolate(x, size=(480, 640), mode='bilinear')
            zeros = torch.zeros((len(x), self.num_output_channels, 80, 640)).cuda()
            x = torch.cat([zeros, x, zeros], dim=2)
        else:
            x = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
            pass

        #print(" Normal after interpolate ", x.shape, " crop? ", self.crop)

        return x


############################################################
#  Loss Functions
############################################################

def compute_rpn_class_loss(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for FG/BG.
	"""

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = (rpn_match == 1).long()

    ## Positive and Negative anchors contribute to the loss,
    ## but neutral anchors (match value = 0) don't.
    indices = torch.nonzero(rpn_match != 0)

    ## Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = rpn_class_logits[indices.data[:, 0], indices.data[:, 1], :]
    anchor_class = anchor_class[indices.data[:, 0], indices.data[:, 1]]

    ## Crossentropy loss
    loss = F.cross_entropy(rpn_class_logits, anchor_class)

    return loss


def compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

	target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
		Uses 0 padding to fill in unsed bbox deltas.
	rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
			   -1=negative, 0=neutral anchor.
	rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
	"""

    ## Squeeze last dim to simplify
    rpn_match = rpn_match.squeeze(2)

    ## Positive anchors contribute to the loss, but negative and
    ## neutral anchors (match value of 0 or -1) don't.
    indices = torch.nonzero(rpn_match == 1)
    ## Pick bbox deltas that contribute to the loss
    rpn_bbox = rpn_bbox[indices.data[:, 0], indices.data[:, 1]]

    ## Trim target bounding box deltas to the same length as rpn_bbox.
    target_bbox = target_bbox[0, :rpn_bbox.size()[0], :]

    ## Smooth L1 loss
    loss = F.smooth_l1_loss(rpn_bbox, target_bbox)

    return loss


def compute_mrcnn_class_loss(target_class_ids, pred_class_logits):
    """Loss for the classifier head of Mask RCNN.

	target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
		padding to fill in the array.
	pred_class_logits: [batch, num_rois, num_classes]
	"""

    ## Loss
    if len(target_class_ids) > 0:
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long())
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_bbox_loss(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

	target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
	target_class_ids: [batch, num_rois]. Integer class IDs.
	pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
	"""

    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the right class_id of each ROI. Get their indicies.
        positive_roi_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = target_class_ids[positive_roi_ix.data].long()
        indices = torch.stack((positive_roi_ix, positive_roi_class_ids), dim=1)

        ## Gather the deltas (predicted and true) that contribute to loss
        target_bbox = target_bbox[indices[:, 0].data, :]
        pred_bbox = pred_bbox[indices[:, 0].data, indices[:, 1].data, :]

        ## Smooth L1 loss
        loss = F.smooth_l1_loss(pred_bbox, target_bbox)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_mask_loss(config, target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

	target_masks: [batch, num_rois, height, width].
		A float32 tensor of values 0 or 1. Uses zero padding to fill array.
	target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
	pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
				with values from 0 to 1.
	"""
    if (target_class_ids > 0).sum() > 0:
        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_masks[indices[:, 0].data, :, :]

        y_pred = pred_masks[indices[:, 0].data, indices[:, 1].data, :, :]

        loss = F.binary_cross_entropy(y_pred, y_true)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_depth_loss_L1(target_depth, pred_depth, thresh=0):
    # TODO: Modify for batch.

    loss = l1LossMask(pred_depth, target_depth,
                      (target_depth > thresh).float())

    return loss


def compute_depth_loss_L2(target_depth, pred_depth, thresh=0):
    # TODO: Modify for batch.

    loss = l2LossMask(pred_depth, target_depth,
                      (target_depth > thresh).float())

    return loss

def compute_depth_loss_berHu(target, pred, thresh=0):
    # TODO: Modify for batch.

    assert pred.dim() == target.dim(), "inconsistent dimensions"
    diff = torch.abs(target - pred)
    valid_mask = (target > thresh).detach()
    diff = diff[valid_mask]
    huber_c = torch.max(diff) * 0.2

    L1_mask = (diff <= huber_c).detach()
    part1 = diff[L1_mask]

    L2_mask = (diff > huber_c).detach()
    diff2 = diff[L2_mask]
    part2 = (diff2 ** 2 + huber_c ** 2) / (2 * huber_c)

    loss = torch.cat((part1, part2)).mean()

    return loss

def compute_depth_loss_edge(target, pred, edges):

    loss = l1LossMask(pred, target, edges)

    return loss

def compute_depth_loss(target_depth, pred_depth, config, edges = None):
    depth_loss = 0
    if config.DEPTH_LOSS == 'L1':
        depth_loss = compute_depth_loss_L1(target_depth[:, 80:560], pred_depth[:, 80:560], config.DEPTH_THRESHOLD)
        #depth_loss = compute_depth_loss_L1(target_depth[:, 40:280], pred_depth[:, 40:280], config.DEPTH_THRESHOLD)
    if config.DEPTH_LOSS == 'L2':
        depth_loss = compute_depth_loss_L1(target_depth[:, 80:560], pred_depth[:, 80:560], config.DEPTH_THRESHOLD)
    if config.DEPTH_LOSS == 'BERHU':
        depth_loss = compute_depth_loss_berHu(target_depth[:, 80:560], pred_depth[:, 80:560], config.DEPTH_THRESHOLD)
    if config.DEPTH_LOSS == 'EDGE':
        edge_loss = compute_depth_loss_edge(target_depth[:, 80:560], pred_depth[:, 80:560], edges[80:560])
        depth_loss = compute_depth_loss_L1(target_depth[:, 80:560], pred_depth[:, 80:560], config.DEPTH_THRESHOLD)
        depth_loss += edge_loss
    if config.DEPTH_LOSS == 'CHAMFER':
        depth_loss = calculate_chamfer_scene(target_depth[:, 80:560], pred_depth[:, 80:560])
    if config.CHAM_LOSS:
        depth_loss += 10*calculate_chamfer_scene(target_depth[:, 80:560], pred_depth[:, 80:560])
    if config.GRAD_LOSS:
        loss_grad = compute_grad_depth_loss(target_depth[:, 80:560], pred_depth[:, 80:560])
        depth_loss += loss_grad

    return depth_loss


def compute_normal_loss(target_normal, pred_normal, config):

    #print(target_plane.shape, pred_plane.shape)
    pred_n = target_normal[:, 80:560]
    target_n = pred_normal[0, :, 80:560]

    #pred_n = pred_n.contiguous().view(-1, 3)
    #target_n = target_n.contiguous().view(-1, 3)

    #loss = F.mse_loss(pred_n, target_n)
    ## L2_Mask_Loss is also possible!

    loss = l1LossMask(pred_n, target_n,
                      (target_n > 0).float())

    return loss

def compute_grad_depth_loss(target_depth, pred_depth):

    pred_depth_batch = pred_depth.view(pred_depth.shape[0], 1, pred_depth.shape[1], pred_depth.shape[2])
    target_depth_batch = target_depth.view(target_depth.shape[0], 1, target_depth.shape[1], target_depth.shape[2])
    grad_real, grad_fake = imgrad_yx(target_depth_batch), imgrad_yx(pred_depth_batch)
    loss = grad_loss(grad_fake, grad_real)

    return loss


def compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                   target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_depth, pred_depth, target_normal,
                   pred_normal, edges):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(config, target_mask, target_class_ids, mrcnn_mask)

    depth_loss = torch.tensor([0], dtype=torch.float32)
    if config.PREDICT_DEPTH:
        depth_loss = compute_depth_loss(target_depth, pred_depth, config)

    normal_loss = torch.tensor([0], dtype=torch.float32)
    if config.PREDICT_PLANE:
        normal_loss = compute_normal_loss(target_normal, pred_normal, config)

    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, normal_loss]


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

	augment: If true, apply random image augmentation. Currently, ony
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
	depth: [height, width, C]
	"""
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    depth = dataset.load_depth(image_id)
    depth = depth / 1000

    shape = image.shape
    # image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    #depth = utils.resize_depth(depth, scale, padding, crop)
    depth, _, _, _, _ = resize_depth_image(
        depth,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    if config.PREDICT_PLANE or config.PREDICT_NORMAL:
        plane = dataset.load_normals(image_id)
        plane, _, _, _, _ = utils.resize_image(
            plane,
            min_dim=config.IMAGE_MIN_DIM,
            min_scale=config.IMAGE_MIN_SCALE,
            max_dim=config.IMAGE_MAX_DIM,
            mode=config.IMAGE_RESIZE_MODE)
    else:
        plane = np.zeros(shape)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            depth = np.fliplr(depth)
            plane = np.fliplr(plane)

    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        # MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
        #				   "Fliplr", "Flipud", "CropAndPad",
        #				   "Affine", "PiecewiseAffine"]

        # def hook(images, augmenter, parents, default):
        #	"""Determines which augmenters to apply to masks."""
        #	return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        depth_shape = depth.shape
        plane_shape = plane.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # depth = det.augment_image(depth, hooks=imgaug.HooksImages(activator=hook))
        # mask = det.augment_image(mask.astype(np.uint8),
        #						 hooks=imgaug.HooksImages(activator=hook))
        mask = det.augment_image(mask.astype(np.uint8))
        depth = det.augment_image(depth.astype(np.float32))
        plane = det.augment_image(plane.astype(np.uint8))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert depth.shape == depth_shape, "Augmentation shouldn't change depth size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        assert plane.shape == plane_shape, "Augmentation shouldn't change plane size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    # if use_mini_mask:
    #	mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # print("Load gt mask shape: ", mask.shape)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)

    return image, image_meta, class_ids, bbox, mask, depth, plane

def load_image_gt_depth(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

	augment: If true, apply random image augmentation. Currently, ony
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
	depth: [height, width, C]
	"""
    # Load image and mask
    image = dataset.load_image(image_id)
    depth = dataset.load_depth(image_id)
    depth = depth / 1000

    shape = image.shape
    # image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    depth = utils.resize_depth(depth, scale, padding, crop)


    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            depth = np.fliplr(depth)

    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        # MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
        #				   "Fliplr", "Flipud", "CropAndPad",
        #				   "Affine", "PiecewiseAffine"]

        # def hook(images, augmenter, parents, default):
        #	"""Determines which augmenters to apply to masks."""
        #	return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        depth_shape = depth.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # depth = det.augment_image(depth, hooks=imgaug.HooksImages(activator=hook))
        # mask = det.augment_image(mask.astype(np.uint8),
        #						 hooks=imgaug.HooksImages(activator=hook))
        depth = det.augment_image(depth.astype(np.float32))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert depth.shape == depth_shape, "Augmentation shouldn't change depth size"
        # Change mask back to bool


    return image, depth


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
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinment() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, config, augmentation=None):
        """A generator that returns images and corresponding target class ids,
			bounding box deltas, and masks.

			dataset: The Dataset object to pick data from
			config: The model config object
			shuffle: If True, shuffles the samples before every epoch
			augment: If True, applies image augmentation to images (currently only
					 horizontal flips are supported)

			Returns a Python generator. Upon calling next() on it, the
			generator returns two lists, inputs and outputs. The containtes
			of the lists differs depending on the received arguments:
			inputs list:
			- images: [batch, H, W, C]
			- image_metas: [batch, size of image meta]
			- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
			- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
			- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
			- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
			- gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
						are those of the image unless use_mini_mask is True, in which
						case they are defined in MINI_MASK_SHAPE.
			- gt_depth: [batch, H, W, DEPTH_CHANNELS] The ground truth depth.

			outputs list: Usually empty in regular training. But if detection_targets
				is True then the outputs list contains target class_ids, bbox deltas,
				and masks.
			"""
        self.b = 0  # batch item index
        self.image_index = -1
        self.image_ids = np.copy(dataset.image_ids)

        self.dataset = dataset
        self.config = config
        self.augmentation = augmentation

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]

        backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                      config.RPN_ANCHOR_RATIOS,
                                                      backbone_shapes,
                                                      config.BACKBONE_STRIDES,
                                                      config.RPN_ANCHOR_STRIDE)

    def __getitem__(self, image_index):
        # Get GT bounding boxes and masks for image.
        image_id = self.image_ids[image_index]
        image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth, gt_normal = \
            load_image_gt(self.dataset, config, image_id, augmentation=self.augmentation,
                          use_mini_mask=False)

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            return None

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, self.anchors,
                                                gt_class_ids, gt_boxes, self.config)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        images = mold_image(image.astype(np.float32), self.config)

        # Convert
        images = torch.from_numpy(images.transpose(2, 0, 1)).float()
        image_metas = torch.from_numpy(image_metas)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        gt_class_ids = torch.from_numpy(gt_class_ids)
        gt_boxes = torch.from_numpy(gt_boxes).float()
        gt_masks = torch.from_numpy(gt_masks.astype(int).transpose(2, 0, 1)).float()

        return images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks

    def __len__(self):
        return self.image_ids.shape[0]


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	random_rois: If > 0 then generate proposals to be used to train the
				 network classifier and mask heads. Useful if training
				 the Mask RCNN part without the RPN.
	batch_size: How many images to return in each call
	detection_targets: If True, generate detection targets (class IDs, bbox
		deltas, and masks). Typically for debugging or visualizations because
		in trainig detection targets are generated by DetectionTargetLayer.
	no_augmentation_sources: Optional. List of sources to exclude for
		augmentation. A source is string that identifies a dataset and is
		defined in the Dataset class.
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The contents
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
	- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
	- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
	- gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
				are those of the image unless use_mini_mask is True, in which
				case they are defined in MINI_MASK_SHAPE.
	- depth: [batch, height, width]. The height and width
			  are those of the image.
	outputs list: Usually empty in regular training. But if detection_targets
		is True then the outputs list contains target class_ids, bbox deltas,
		and masks.
	"""
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth, gt_plane = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                  use_mini_mask=config.USE_MINI_MASK)
            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth, gt_plane = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                  use_mini_mask=config.USE_MINI_MASK, augmentation=augmentation)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                batch_depth = np.zeros(
                    (batch_size,) + gt_depth.shape, dtype=np.float32)
                batch_plane = np.zeros(
                    (batch_size,) + gt_plane.shape, dtype=np.float32)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_depth[b] = gt_depth.astype(np.float32)
            batch_plane[b] = gt_plane.astype(np.float32)
            b += 1

            # Batch full?
            if b >= batch_size:
                # depth = np.zeros(1)

                yield [torch.from_numpy(batch_images.transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_image_meta), torch.from_numpy(batch_rpn_match),
                       torch.from_numpy(batch_rpn_bbox.astype(np.float32)), torch.from_numpy(batch_gt_class_ids),
                       torch.from_numpy(batch_gt_boxes.astype(np.float32)),
                       torch.from_numpy(batch_gt_masks.astype(np.float32).transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_depth.astype(np.float32)),
                       torch.from_numpy(batch_plane.astype(np.float32).transpose(0, 3, 1, 2))]

                # inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                #          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                # outputs = []

                # yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def data_get_withid(dataset, config, image_id, shuffle=False, augment=False,
                    random_rois=0, batch_size=1, detection_targets=False,
                    no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	random_rois: If > 0 then generate proposals to be used to train the
				 network classifier and mask heads. Useful if training
				 the Mask RCNN part without the RPN.
	batch_size: How many images to return in each call
	detection_targets: If True, generate detection targets (class IDs, bbox
		deltas, and masks). Typically for debugging or visualizations because
		in trainig detection targets are generated by DetectionTargetLayer.
	no_augmentation_sources: Optional. List of sources to exclude for
		augmentation. A source is string that identifies a dataset and is
		defined in the Dataset class.
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The contents
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
	- rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
	- gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
	- gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
	- gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
				are those of the image unless use_mini_mask is True, in which
				case they are defined in MINI_MASK_SHAPE.
	- depth: [batch, height, width]. The height and width
			  are those of the image.
	outputs list: Usually empty in regular training. But if detection_targets
		is True then the outputs list contains target class_ids, bbox deltas,
		and masks.
	"""
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    try:
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        image_index = image_id
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)

        # Get GT bounding boxes and masks for image.
        image_id = image_ids[image_index]

        # If the image source is not to be augmented pass None as augmentation
        if dataset.image_info[image_id]['source'] in no_augmentation_sources:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)
        else:
            image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)

        # RPN Targets
        rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                gt_class_ids, gt_boxes, config)

        # Init batch arrays
        if b == 0:
            batch_image_meta = np.zeros(
                (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
            batch_rpn_match = np.zeros(
                [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros(
                [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
            batch_images = np.zeros(
                (batch_size,) + image.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_boxes = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
            batch_gt_masks = np.zeros(
                (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                 config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
            batch_depth = np.zeros(
                (batch_size,) + gt_depth.shape, dtype=np.float32)

        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = mold_image(image.astype(np.float32), config)
        batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
        batch_depth[b] = gt_depth.astype(np.float32)
        b += 1

        # Batch full?
        if b >= batch_size:
            # depth = np.zeros(1)

            return [torch.from_numpy(batch_images.transpose(0, 3, 1, 2)),
                    torch.from_numpy(batch_image_meta), torch.from_numpy(batch_rpn_match),
                    torch.from_numpy(batch_rpn_bbox.astype(np.float32)), torch.from_numpy(batch_gt_class_ids),
                    torch.from_numpy(batch_gt_boxes.astype(np.float32)),
                    torch.from_numpy(batch_gt_masks.astype(np.float32).transpose(0, 3, 1, 2)),
                    torch.from_numpy(batch_depth.astype(np.float32))]

            # inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
            #          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            # outputs = []

            # yield inputs, outputs

            # start a new batch
            b = 0
    except (GeneratorExit, KeyboardInterrupt):
        raise
    except:
        # Log it and skip the image
        logging.exception("Error processing image {}".format(
            dataset.image_info[image_id]))
        error_count += 1
        if error_count > 5:
            raise


def data_generator_onlydepth(dataset, config, shuffle=True, augment=False, augmentation=None,
                             random_rois=0, batch_size=1, detection_targets=False,
                             no_augmentation_sources=None):
    """A generator that returns images and corresponding target class ids,
	bounding box deltas, and masks.
	dataset: The Dataset object to pick data from
	config: The model config object
	shuffle: If True, shuffles the samples before every epoch
	augment: (deprecated. Use augmentation instead). If true, apply random
		image augmentation. Currently, only horizontal flipping is offered.
	augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
		For example, passing imgaug.augmenters.Fliplr(0.5) flips images
		right/left 50% of the time.
	batch_size: How many images to return in each call
	no_augmentation_sources: Optional. List of sources to exclude for
		augmentation. A source is string that identifies a dataset and is
		defined in the Dataset class.
	Returns a Python generator. Upon calling next() on it, the
	generator returns two lists, inputs and outputs. The contents
	of the lists differs depending on the received arguments:
	inputs list:
	- images: [batch, H, W, C]
	- image_meta: [batch, (meta data)] Image details. See compose_image_meta()
	- depth: [batch, height, width]. The height and width
			  are those of the image.
	"""
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth, gt_normal = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                  use_mini_mask=False)

            else:
                image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_depth, gt_normal = \
                    load_image_gt(dataset, config, image_id, augment=augment, augmentation=augmentation,
                                  use_mini_mask=False)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_depth = np.zeros(
                    (batch_size,) + gt_depth.shape, dtype=np.float32)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_depth[b] = gt_depth.astype(np.float32)
            b += 1

            # Batch full?
            if b >= batch_size:
                # depth = np.zeros(1)

                yield [torch.from_numpy(batch_images.transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_image_meta),
                       torch.from_numpy(batch_depth.astype(np.float32))]

                # inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                #          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                # outputs = []

                # yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


############################################################
#  MaskRCNN Class for Depth
############################################################

class MaskRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality for
	bounding box prediction, mask prediction, object classification,
	and depth prediction.
	"""

    def __init__(self, config, model_dir='checkpoints'):
        """
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights
		"""
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
		"""

        ## Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ## Build the shared convolutional layers.
        ## Bottom-up Layers
        ## Returns a list of the last layers of each stage, 5 in total.
        ## Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        C1, C2, C3, C4, C5 = resnet.stages()

        ## Top-down Layers
        ## TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)

        ## Generate Anchors
        backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                backbone_shapes,
                                                                                config.BACKBONE_STRIDES,
                                                                                config.RPN_ANCHOR_STRIDE)).float(),
                                requires_grad=False)
        if self.config.GPU_COUNT:
            self.anchors = self.anchors.cuda()

        ## RPN
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)

        ## FPN Classifier
        self.debug = False
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES,
                                     debug=self.debug)

        ## FPN Mask
        self.mask = Mask(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        if self.config.PREDICT_DEPTH:
            if self.config.PREDICT_BOUNDARY:
                self.depth = Depth(num_output_channels=3)
            else:
                self.depth = Depth(num_output_channels=1)
                pass
            pass

        if self.config.PREDICT_PLANE:
            self.plane = Depth(num_output_channels=3)
            pass

        ## Fix batch norm layers

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
		"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
		the given regular expression.
		"""

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

		model_path: If None, or a format different from what this code uses
			then set a new log directory and start epochs from 0. Otherwise,
			extract the log directory and the epoch counter from the file
			name.
		"""

        ## Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        ## If we have a model path with date and epochs use them
        if model_path:
            ## Continue from we left of. Get epoch and date from the file name
            ## A sample model path might look like:
            ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        ## Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        ## Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
		model directory.
		Returns:
			log_dir: The directory where events and weights are saved
			checkpoint_path: the path to the last checkpoint file
		"""
        ## Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        ## Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        ## Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, iscoco=False):
        """Modified version of the correspoding Keras function with
		the addition of multi-GPU support and the ability to exclude
		some layers from loading.
		exlude: list of layer names to excluce
		"""
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            if iscoco:
                state = self.state_dict()
                state_dict = {k: v for k, v in state_dict.items() if
                              k in state.keys() and 'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k and 'fpn.C1.0' not in k and 'classifier.conv1' not in k}
                state.update(state_dict)
                self.load_state_dict(state)
            else:
                try:
                    self.load_state_dict(state_dict, strict=False)
                except:
                    # print('load only base model')
                    try:
                        state_dict = {k: v for k, v in state_dict.items() if
                                      'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k}
                        state = self.state_dict()
                        state.update(state_dict)
                        self.load_state_dict(state)
                    except:
                        # print('change input dimension')
                        state_dict = {k: v for k, v in state_dict.items() if
                                      'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k and 'fpn.C1.0' not in k and 'classifier.conv1' not in k}
                        state = self.state_dict()
                        state.update(state_dict)
                        self.load_state_dict(state)
                        pass
                    pass
        else:
            # print("Weight file not found ...")
            exit(1)
        ## Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def detect(self, images, mold_image=True, image_metas=None):
        """Runs the detection pipeline.

		images: List of images, potentially of different sizes.

		Returns a list of dicts, one dict per image. The dict contains:
		rois: [N, (y1, x1, y2, x2)] detection bounding boxes
		class_ids: [N] int class IDs
		scores: [N] float probability scores for the class IDs
		masks: [H, W, N] instance binary masks
		"""

        ## Mold inputs to format expected by the neural network
        if mold_image:
            molded_images, image_metas, windows = self.mold_inputs(self.config, images)
        else:
            molded_images = images
            windows = [(0, 0, images.shape[1], images.shape[2]) for _ in range(len(images))]
            pass

        ## Convert images to torch tensor
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()

        ## To GPU
        if self.config.GPU_COUNT:
            molded_images = molded_images.cuda()

        ## Wrap in variable
        # molded_images = Variable(molded_images, volatile=True)

        ## Run object detection
        detections, mrcnn_mask, depth_np, plane_np = self.predict([molded_images, image_metas], mode='inference')

        if len(detections[0]) == 0:
            return [{'rois': [], 'class_ids': [], 'scores': [], 'masks': [], 'parameters': []}]

        ## Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        ## Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(self.config, detections[i], mrcnn_mask[i],
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def predict(self, input, mode, use_nms=1, use_refinement=False, return_feature_map=False):
        molded_images = input[0]
        image_metas = input[1]

        if mode == 'inference':
            self.eval()
        elif 'training' in mode:
            self.train()

            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        ## Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        ## Note that P6 is used in RPN, but not in the classifier heads.

        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        feature_maps = [feature_map for index, feature_map in enumerate(rpn_feature_maps[::-1])]
        if self.config.PREDICT_DEPTH:
            depth_np = self.depth(feature_maps)
            if self.config.PREDICT_BOUNDARY:
                boundary = depth_np[:, 1:]
                depth_np = depth_np[:, 0]
            else:
                depth_np = depth_np.squeeze(1)
                pass
        else:
            depth_np = torch.ones((1, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)).cuda()
            pass

        if self.config.PREDICT_PLANE:
            plane_np = self.plane(feature_maps)
            plane_np = plane_np.squeeze(0)
        else:
            plane_np = torch.ones((3, self.config.IMAGE_MAX_DIM, self.config.IMAGE_MAX_DIM)).cuda()
            pass

        ## Loop through pyramid layers
        layer_outputs = []  ## list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(self.rpn(p))

        ## Concatenate layer outputs
        ## Convert from list of lists of level outputs to list of lists
        ## of outputs across levels.
        ## e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox = outputs

        ## Generate proposals
        ## Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        ## and zero padded.
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if 'training' in mode and use_refinement == False \
            else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox],
                                  proposal_count=proposal_count,
                                  nms_threshold=self.config.RPN_NMS_THRESHOLD,
                                  anchors=self.anchors,
                                  config=self.config)

        if mode == 'inference':
            ## Network Heads
            ## Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps,
                                                                          rpn_rois)

            ## Detections
            ## output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = detection_layer_(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)

            if len(detections) == 0:
                return [[]], [[]], depth_np
            ## Convert boxes to normalized coordinates
            ## TODO: let DetectionLayer return normalized coordinates to avoid
            ##       unnecessary conversions
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            detection_boxes = detections[:, :4] / scale

            ## Add back batch dimension
            detection_boxes = detection_boxes.unsqueeze(0)

            ## Create masks for detections
            mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, detection_boxes)

            ## Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            return [detections, mrcnn_mask, depth_np, plane_np]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]

            ## Normalize coordinates
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT:
                scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            ## Generate detection targets
            ## Subsamples proposals and generates target outputs for training
            ## Note that proposal class IDs, gt_boxes, and gt_masks are zero
            ## padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = \
                detection_target_layer_(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_parameters = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_parameters = mrcnn_parameters.cuda()
            else:
                ## Network Heads
                ## Proposal classifier and BBox regressor heads
                # print([maps.shape for maps in mrcnn_feature_maps], target_parameters.shape)
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)

                # print("feature maps: ", len(mrcnn_feature_maps))
                # print("feature maps: ", mrcnn_feature_maps[0].shape)
                # print("ROIs: ", len(rois))
                # print("ROIs: ", rois[0].shape)

                ## Create masks for detections
                mrcnn_mask, _ = self.mask(mrcnn_feature_maps, rois)

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, rois, depth_np, plane_np]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers, depth_weight=1,
                    optim_choice="SGD"):
        """Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heaads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  3+: Train Resnet stage 3 and up
			  4+: Train Resnet stage 4 and up
			  5+: Train Resnet stage 5 and up
		"""

        # Pre-defined layer regular expressions
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

        # Data generators
        # train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
        # val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch + 1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]

        if optim_choice == "ADAM":
            optimizer = optim.Adam([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate)
        else:
            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth = self.train_epoch(
                train_generator, optimizer, self.config.STEPS_PER_EPOCH, depth_weight)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_depth = self.valid_epoch(
                val_generator, self.config.VALIDATION_STEPS, depth_weight)

            # Statistics
            self.loss_history.append(
                [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth])
            self.val_loss_history.append(
                [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
                 val_loss_mrcnn_mask, val_loss_depth])
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/RPN_Class', loss_rpn_class, epoch)
            writer.add_scalar('Train/RPN_BBOX', loss_rpn_bbox, epoch)
            writer.add_scalar('Train/MRCNN_Class', loss_mrcnn_class, epoch)
            writer.add_scalar('Train/MRCNN_BBOX', loss_mrcnn_bbox, epoch)
            writer.add_scalar('Train/MRCNN_Mask', loss_mrcnn_mask, epoch)
            writer.add_scalar('Train/Depth', loss_depth, epoch)

            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/RPN_Class', val_loss_rpn_class, epoch)
            writer.add_scalar('Val/RPN_BBOX', val_loss_rpn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Class', val_loss_mrcnn_class, epoch)
            writer.add_scalar('Val/MRCNN_BBOX', val_loss_mrcnn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Mask', val_loss_mrcnn_mask, epoch)
            writer.add_scalar('Val/Depth', val_loss_depth, epoch)

            # Save model
            if epoch % 25 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_model2(self, train_dataset, val_dataset, learning_rate, epochs, layers, depth_weight=1,
                     optim_choice="SGD"):
        """Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heaads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  3+: Train Resnet stage 3 and up
			  4+: Train Resnet stage 4 and up
			  5+: Train Resnet stage 5 and up
		"""

        # Pre-defined layer regular expressions
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

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, augment=True, batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True, augment=True, batch_size=self.config.BATCH_SIZE)

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch + 1, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]

        if optim_choice == "ADAM":
            optimizer = optim.Adam([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate)
        else:
            optimizer = optim.SGD([
                {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
                {'params': trainables_only_bn}
            ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth, loss_plane = self.train_epoch(
                train_generator, optimizer, self.config.STEPS_PER_EPOCH, depth_weight)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask, val_loss_depth, val_loss_plane = self.valid_epoch(
                val_generator, self.config.VALIDATION_STEPS, depth_weight)

            # Statistics
            self.loss_history.append(
                [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, loss_depth, loss_plane])
            self.val_loss_history.append(
                [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
                 val_loss_mrcnn_mask, val_loss_depth, val_loss_plane])
            visualize.plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/RPN_Class', loss_rpn_class, epoch)
            writer.add_scalar('Train/RPN_BBOX', loss_rpn_bbox, epoch)
            writer.add_scalar('Train/MRCNN_Class', loss_mrcnn_class, epoch)
            writer.add_scalar('Train/MRCNN_BBOX', loss_mrcnn_bbox, epoch)
            writer.add_scalar('Train/MRCNN_Mask', loss_mrcnn_mask, epoch)
            writer.add_scalar('Train/Depth', loss_depth, epoch)
            writer.add_scalar('Train/Plane', loss_plane, epoch)

            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/RPN_Class', val_loss_rpn_class, epoch)
            writer.add_scalar('Val/RPN_BBOX', val_loss_rpn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Class', val_loss_mrcnn_class, epoch)
            writer.add_scalar('Val/MRCNN_BBOX', val_loss_mrcnn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Mask', val_loss_mrcnn_mask, epoch)
            writer.add_scalar('Val/Depth', val_loss_depth, epoch)
            writer.add_scalar('Val/Plane', val_loss_plane, epoch)

            # Save model
            if epoch % 10 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps, depth_weight=1, plane_weight=0.001):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_depth_sum = 0
        loss_plane_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_depths = inputs[7]
            gt_plane = inputs[8]

            # image_metas as numpy array
            # image_metas = image_metas.numpy()

            gt_masks = torch.FloatTensor(gt_masks)

            # Wrap in variables
            images = Variable(images)
            rpn_match = Variable(rpn_match)
            rpn_bbox = Variable(rpn_bbox)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            gt_depths = Variable(gt_depths)
            gt_plane = Variable(gt_plane)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_depths = gt_depths.cuda()
                gt_plane = gt_plane.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, rois, pred_depth, pred_plane = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, plane_loss = compute_losses(
                self.config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, gt_depths, pred_depth, gt_plane, pred_plane)

            if self.config.GPU_COUNT:
                depth_loss = depth_loss.cuda()

            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + depth_weight * depth_loss + plane_weight * plane_loss

            # Added this due to a weird error.
            # loss = Variable(loss, requires_grad=True)

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            # Progress
            if step % 200 == 0:
                printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                 suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - depth_loss: {:.5f} - plane_loss: {:.5f}".format(
                                     loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                     rpn_bbox_loss.data.cpu().item(),
                                     mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                     mrcnn_mask_loss.data.cpu().item(), depth_loss.data.cpu().item(), plane_loss.data.cpu().item()), length=10)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item() / steps
            loss_depth_sum += depth_loss.data.cpu().item() / steps
            loss_plane_sum += plane_loss.data.cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_depth_sum, loss_plane_sum

    def valid_epoch(self, datagenerator, steps, depth_weight=1, plane_weight=0.001):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_depth_sum = 0
        loss_plane_sum = 0

        with torch.no_grad():
            for inputs in datagenerator:
                images = inputs[0]
                image_metas = inputs[1]
                rpn_match = inputs[2]
                rpn_bbox = inputs[3]
                gt_class_ids = inputs[4]
                gt_boxes = inputs[5]
                gt_masks = inputs[6]
                gt_depths = inputs[7]
                gt_plane = inputs[8]

                # image_metas as numpy array
                # image_metas = image_metas.numpy()

                gt_masks = torch.FloatTensor(gt_masks)

                # Wrap in variables
                images = Variable(images)
                rpn_match = Variable(rpn_match)
                rpn_bbox = Variable(rpn_bbox)
                gt_class_ids = Variable(gt_class_ids)
                gt_boxes = Variable(gt_boxes)
                gt_masks = Variable(gt_masks)
                gt_depths = Variable(gt_depths)
                gt_plane = Variable(gt_plane)

                # To GPU
                if self.config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()
                    gt_depths = gt_depths.cuda()
                    gt_plane = gt_plane.cuda()

                # Run object detection
                rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, \
                target_mask, mrcnn_mask, rois, pred_depth, pred_plane = \
                    self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

                if not target_class_ids.size():
                    continue

                # Compute losses
                rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, plane_loss = compute_losses(
                    self.config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                    mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, gt_depths, pred_depth, gt_plane, pred_plane)

                if self.config.GPU_COUNT:
                    depth_loss = depth_loss.cuda()

                loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss + depth_weight * depth_loss + plane_loss*plane_weight

                # Progress
                if step % 100 == 0:
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - depth_loss: {:.5f} - plane_loss: {:.5f}".format(
                                         loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                         rpn_bbox_loss.data.cpu().item(),
                                         mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                         mrcnn_mask_loss.data.cpu().item(), depth_loss.data.cpu().item(), plane_loss.data.cpu().item()), length=10)

                # Statistics
                loss_sum += loss.data.cpu().item() / steps
                loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
                loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
                loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item() / steps
                loss_depth_sum += depth_loss.data.cpu().item() / steps
                loss_plane_sum += plane_loss.data.cpu().item() / steps

                # Break after 'steps' steps
                if step == steps - 1:
                    break
                step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum, loss_depth_sum, loss_plane_sum

    def mold_inputs(self, config, images):
        """Takes a list of images and modifies them to the format expected
		as an input to the neural network.
		images: List of image matricies [height,width,depth]. Images can have
			different sizes.

		Returns 3 Numpy matricies:
		molded_images: [N, h, w, 3]. Images resized and normalized.
		image_metas: [N, length of meta data]. Details about each image.
		windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
			original image (padding excluded).
		"""
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            ## Resize image to fit the model expected size
            ## TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                mode=self.config.IMAGE_RESIZE_MODE)
            # padding=config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, config)
            ## Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([config.NUM_CLASSES], dtype=np.int32))
            ## Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        ## Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, config, detections, mrcnn_mask, image_shape, window, debug=False):
        """Reformats the detections of one image from the format of the neural
		network output to a format suitable for use in the rest of the
		application.

		detections: [N, (y1, x1, y2, x2, class_id, score)]
		mrcnn_mask: [N, height, width, num_classes]
		image_shape: [height, width, depth] Original size of the image before resizing
		window: [y1, x1, y2, x2] Box in the image where the real image is
				excluding the padding.

		Returns:
		boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
		class_ids: [N] Integer class IDs for each bounding box
		scores: [N] Float probability scores of the class_id
		masks: [height, width, num_instances] Instance masks
		"""
        ## How many detections do we have?
        ## Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        ## Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        if config.GLOBAL_MASK:
            masks = mrcnn_mask[np.arange(N), :, :, 0]
        else:
            masks = mrcnn_mask[np.arange(N), :, :, class_ids]
            pass

        ## Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  ## y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        ## Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        if debug:
            # print(masks.shape, boxes.shape)
            for maskIndex, mask in enumerate(masks):
                # print(maskIndex, boxes[maskIndex].astype(np.int32))
                cv2.imwrite('test/local_mask_' + str(maskIndex) + '.png', (mask * 255).astype(np.uint8))
                continue

        ## Filter out detections with zero area. Often only happens in early
        ## stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        ## Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            ## Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        if debug:
            # print(full_masks.shape)
            for maskIndex in range(full_masks.shape[2]):
                cv2.imwrite('test/full_mask_' + str(maskIndex) + '.png',
                            (full_masks[:, :, maskIndex] * 255).astype(np.uint8))
                continue
            pass
        return boxes, class_ids, scores, full_masks


############################################################
#  DepthCNN
############################################################


class DepthCNN(nn.Module):
    """Encapsulates the Depth CNN model functionality for
	depth prediction from ResNet and FPN features.
	"""

    def __init__(self, config, model_dir='checkpoints'):
        """
		config: A Sub-class of the Config class
		model_dir: Directory to save training logs and trained weights
		"""
        super(DepthCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def build(self, config):
        """Build Mask R-CNN architecture.
		"""

        ## Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        ## Build the shared convolutional layers.
        ## Bottom-up Layers
        ## Returns a list of the last layers of each stage, 5 in total.
        ## Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet50", stage5=True, numInputChannels=config.NUM_INPUT_CHANNELS)
        C1, C2, C3, C4, C5 = resnet.stages()

        ## Top-down Layers
        ## TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256, bilinear_upsampling=self.config.BILINEAR_UPSAMPLING)

        self.depth = Depth(num_output_channels=1)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def initialize_weights(self):
        """Initialize model weights.
		"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
		the given regular expression.
		"""

        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable:
                param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

		model_path: If None, or a format different from what this code uses
			then set a new log directory and start epochs from 0. Otherwise,
			extract the log directory and the epoch counter from the file
			name.
		"""

        ## Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        ## If we have a model path with date and epochs use them
        if model_path:
            ## Continue from we left of. Get epoch and date from the file name
            ## A sample model path might look like:
            ## /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))

        ## Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        ## Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{:04d}")

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
		model directory.
		Returns:
			log_dir: The directory where events and weights are saved
			checkpoint_path: the path to the last checkpoint file
		"""
        ## Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        ## Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        ## Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
		the addition of multi-GPU support and the ability to exclude
		some layers from loading.
		exlude: list of layer names to excluce
		"""
        if os.path.exists(filepath):
            state_dict = torch.load(filepath)
            try:
                self.load_state_dict(state_dict, strict=False)
            except:
                # print('load only base model')
                try:
                    state_dict = {k: v for k, v in state_dict.items() if
                                  'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                except:
                    # print('change input dimension')
                    state_dict = {k: v for k, v in state_dict.items() if
                                  'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k and 'fpn.C1.0' not in k and 'classifier.conv1' not in k}
                    state = self.state_dict()
                    state.update(state_dict)
                    self.load_state_dict(state)
                    pass
                pass
        else:
            # print("Weight file not found ...")
            exit(1)
        ## Update the log directory
        self.set_log_dir(filepath)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def predict(self, input, mode):
        molded_images = input[0]

        if mode == 'inference':
            self.eval()
        elif 'training' in mode:
            self.train()

            ## Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)

        ## Feature extraction
        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)
        ## Note that P6 is used in RPN, but not in the classifier heads.

        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]

        feature_maps = [feature_map for index, feature_map in enumerate(rpn_feature_maps[::-1])]
        # print("feature maps size: ", feature_maps[0].shape, feature_maps[1].shape, feature_maps[2].shape, feature_maps[3].shape, feature_maps[4].shape)
        depth_np = self.depth(feature_maps)
        depth_np = depth_np.squeeze(1)

        return [depth_np]

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers="all"):
        """Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heaads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  3+: Train Resnet stage 3 and up
			  4+: Train Resnet stage 4 and up
			  5+: Train Resnet stage 5 and up
		"""

        # Pre-defined layer regular expressions

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

        # Data generators
        # train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4)
        # val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=4)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss_depth = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss_depth = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss_depth])
            self.val_loss_history.append([val_loss_depth])

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss_depth, epoch)
            writer.add_scalar('Val/Loss', val_loss_depth, epoch)

            visualize.plot_depth_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            if epoch % 25 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    # Uses a maskrcnn compatible dataset, loads only depth maps.
    def train_model2(self, train_dataset, val_dataset, learning_rate, epochs, layers="all", augmentation=None):
        """Train the model.
		train_dataset, val_dataset: Training and validation Dataset objects.
		learning_rate: The learning rate to train with
		epochs: Number of training epochs. Note that previous training epochs
				are considered to be done alreay, so this actually determines
				the epochs to train in total rather than in this particaular
				call.
		layers: Allows selecting wich layers to train. It can be:
			- A regular expression to match layer names to train
			- One of these predefined values:
			  heaads: The RPN, classifier and mask heads of the network
			  all: All the layers
			  3+: Train Resnet stage 3 and up
			  4+: Train Resnet stage 4 and up
			  5+: Train Resnet stage 5 and up
		"""

        # Pre-defined layer regular expressions

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

        # Data generators
        train_generator = data_generator_onlydepth(train_dataset, self.config, shuffle=True, augment=False, batch_size=self.config.BATCH_SIZE, augmentation=augmentation)
        val_generator = data_generator_onlydepth(val_dataset, self.config, shuffle=True, augment=False, batch_size=self.config.BATCH_SIZE, augmentation=augmentation)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))
            print("Epoch: ", epoch)

            # Training

            loss_depth = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss_depth = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss_depth])
            self.val_loss_history.append([val_loss_depth])

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss_depth, epoch)
            writer.add_scalar('Val/Loss', val_loss_depth, epoch)

            visualize.plot_depth_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            if epoch % 1 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    # Uses the load_image_gt_depth. Works for SUNCG dataset.
    def train_model3(self, train_dataset, val_dataset, learning_rate, epochs, layers="all"):
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

        # Data generators
        train_generator = data_generator_onlydepth(train_dataset, self.config, shuffle=True, augment=True, batch_size=1)
        val_generator = data_generator_onlydepth(val_dataset, self.config, shuffle=True, augment=True, batch_size=1)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss_depth = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss_depth = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss_depth])
            self.val_loss_history.append([val_loss_depth])

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss_depth, epoch)
            writer.add_scalar('Val/Loss', val_loss_depth, epoch)

            visualize.plot_depth_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            if epoch % 25 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    ## Uses a maskrcnn compatible dataset, loads all values eg. masks, boxes.
    def train_model4(self, train_dataset, val_dataset, learning_rate, epochs, layers="all"):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions

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

        # Data generators
        # train_set = Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE,
                                                      shuffle=True, num_workers=4)
        # val_set = Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True,
                                                    num_workers=4)

        # Optimizer object
        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if
                            param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if
                              param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss_depth = self.train_epoch2(train_generator, optimizer, self.config.STEPS_PER_EPOCH)

            # Validation
            val_loss_depth = self.valid_epoch2(val_generator, self.config.VALIDATION_STEPS)

            # Statistics
            self.loss_history.append([loss_depth])
            self.val_loss_history.append([val_loss_depth])

            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            if os.path.exists(self.log_dir):
                writer = SummaryWriter(self.log_dir + '/log/')

            writer.add_scalar('Train/Loss', loss_depth, epoch)
            writer.add_scalar('Val/Loss', val_loss_depth, epoch)

            visualize.plot_depth_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)

            # Save model
            if epoch % 25 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            edges = inputs[1]
            gt_depths = inputs[2]

            # Wrap in variables
            images = Variable(images)
            edges = Variable(edges)
            gt_depths = Variable(gt_depths)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                gt_depths = gt_depths.cuda()
                edges = edges.cuda()

            # Run depth detection
            pred_depth = self.predict([images], mode='training')[0]

            # print("gt shape: ", gt_depths.shape, " pred shape: ", pred_depth.shape)

            # Compute losses
            loss = compute_depth_loss(gt_depths, pred_depth, self.config, edges[0])

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            # Progress
            if step % 100 == 0:
                printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                 suffix="Complete depth - loss: {:.5f} ".format(
                                     loss.data.cpu().item()), length=10)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum

    # Uses maskrcnn dataset loader, loads masks.
    def train_epoch2(self, datagenerator, optimizer, steps):
        batch_count = 0
        loss_sum = 0
        step = 0

        optimizer.zero_grad()

        for inputs in datagenerator:
            batch_count += 1

            images = inputs[0]
            image_metas = inputs[1]
            rpn_match = inputs[2]
            rpn_bbox = inputs[3]
            gt_class_ids = inputs[4]
            gt_boxes = inputs[5]
            gt_masks = inputs[6]
            gt_depths = inputs[7]

            # image_metas as numpy array
            image_metas = image_metas.numpy()

            # Wrap in variables
            images = Variable(images)
            rpn_match = Variable(rpn_match)
            rpn_bbox = Variable(rpn_bbox)
            gt_class_ids = Variable(gt_class_ids)
            gt_boxes = Variable(gt_boxes)
            gt_masks = Variable(gt_masks)
            gt_depths = Variable(gt_depths)

            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_depths = gt_depths.cuda()

            # Run depth detection
            pred_depth = self.predict([images], mode='training')[0]

            # print("gt shape: ", gt_depths.shape, " pred shape: ", pred_depth.shape)

            # Compute losses
            if self.config.CHAM_COMBINE:
                chamf, l1 = chamfer_L1_combined_loss(images, gt_depths, pred_depth, gt_masks, gt_boxes, gt_class_ids)
            else:
                chamf = calculate_chamfer_masked(images, gt_depths, pred_depth, gt_masks, gt_boxes, gt_class_ids).float().cuda()
                #print("pred depth shape: ", pred_depth.shape, gt_depths.shape)
                l1 = compute_depth_loss_L1(pred_depth[:, 80:560], gt_depths[:, 80:560], self.config.DEPTH_THRESHOLD)

            loss = chamf*10+l1

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            # Progress
            if step % 100 == 0:
                printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                 suffix="Complete depth - loss: {:.5f} ".format(
                                     loss.data.cpu().item()), length=10)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum

    def valid_epoch(self, datagenerator, steps):

        step = 0
        loss_sum = 0

        with torch.no_grad():
            for inputs in datagenerator:
                images = inputs[0]
                edges = inputs[1]
                gt_depths = inputs[2]

                # Wrap in variables
                images = Variable(images)
                edges = Variable(edges)
                gt_depths = Variable(gt_depths)

                # To GPU
                if self.config.GPU_COUNT:
                    images = images.cuda()
                    gt_depths = gt_depths.cuda()
                    edges = edges.cuda()

                # Run object detection
                pred_depth = self.predict([images], mode='training')[0]

                # Compute losses
                loss = compute_depth_loss(gt_depths, pred_depth, self.config, edges[0])

                # Progress
                if step % 100 == 0:
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete depth - loss: {:.5f} ".format(
                                         loss.data.cpu().item()), length=10)

                # Statistics
                loss_sum += loss.data.cpu().item() / steps

                # Break after 'steps' steps
                if step == steps - 1:
                    break
                step += 1

        return loss_sum

    def valid_epoch2(self, datagenerator, steps):

        step = 0
        loss_sum = 0

        with torch.no_grad():
            for inputs in datagenerator:
                images = inputs[0]
                image_metas = inputs[1]
                rpn_match = inputs[2]
                rpn_bbox = inputs[3]
                gt_class_ids = inputs[4]
                gt_boxes = inputs[5]
                gt_masks = inputs[6]
                gt_depths = inputs[7]

                # image_metas as numpy array
                image_metas = image_metas.numpy()

                # Wrap in variables
                images = Variable(images)
                rpn_match = Variable(rpn_match)
                rpn_bbox = Variable(rpn_bbox)
                gt_class_ids = Variable(gt_class_ids)
                gt_boxes = Variable(gt_boxes)
                gt_masks = Variable(gt_masks)
                gt_depths = Variable(gt_depths)

                if self.config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()
                    gt_depths = gt_depths.cuda()

                # Run object detection
                pred_depth = self.predict([images], mode='training')[0]

                # Compute losses
                if self.config.CHAM_COMBINE:
                    chamf, l1 = chamfer_L1_combined_loss(images, gt_depths, pred_depth, gt_masks, gt_boxes,
                                                         gt_class_ids)
                else:
                    chamf = calculate_chamfer_masked(images, gt_depths, pred_depth, gt_masks, gt_boxes,
                                                     gt_class_ids).float().cuda()
                    l1 = compute_depth_loss_L1(pred_depth[:, 80:560], gt_depths[:, 80:560], self.config.DEPTH_THRESHOLD)

                loss = chamf * 10 + l1

                # Progress
                if step % 100 == 0:
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete depth - loss: {:.5f} ".format(
                                         loss.data.cpu().item()), length=10)

                # Statistics
                loss_sum += loss.data.cpu().item() / steps

                # Break after 'steps' steps
                if step == steps - 1:
                    break
                step += 1

        return loss_sum

    def mold_inputs(self, config, images):
        """Takes a list of images and modifies them to the format expected
		as an input to the neural network.
		images: List of image matricies [height,width,depth]. Images can have
			different sizes.

		Returns 3 Numpy matricies:
		molded_images: [N, h, w, 3]. Images resized and normalized.
		image_metas: [N, length of meta data]. Details about each image.
		windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
			original image (padding excluded).
		"""
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            ## Resize image to fit the model expected size
            ## TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                padding=config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, config)
            ## Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([config.NUM_CLASSES], dtype=np.int32))
            ## Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        ## Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, config, detections, mrcnn_mask, image_shape, window, debug=False):
        """Reformats the detections of one image from the format of the neural
		network output to a format suitable for use in the rest of the
		application.

		detections: [N, (y1, x1, y2, x2, class_id, score)]
		mrcnn_mask: [N, height, width, num_classes]
		image_shape: [height, width, depth] Original size of the image before resizing
		window: [y1, x1, y2, x2] Box in the image where the real image is
				excluding the padding.

		Returns:
		boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
		class_ids: [N] Integer class IDs for each bounding box
		scores: [N] Float probability scores of the class_id
		masks: [height, width, num_instances] Instance masks
		"""
        ## How many detections do we have?
        ## Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        ## Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        if config.GLOBAL_MASK:
            masks = mrcnn_mask[np.arange(N), :, :, 0]
        else:
            masks = mrcnn_mask[np.arange(N), :, :, class_ids]
            pass

        ## Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  ## y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        ## Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        if debug:
            # print(masks.shape, boxes.shape)
            for maskIndex, mask in enumerate(masks):
                # print(maskIndex, boxes[maskIndex].astype(np.int32))
                cv2.imwrite('test/local_mask_' + str(maskIndex) + '.png', (mask * 255).astype(np.uint8))
                continue

        ## Filter out detections with zero area. Often only happens in early
        ## stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        ## Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            ## Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty((0,) + masks.shape[1:3])

        if debug:
            # print(full_masks.shape)
            for maskIndex in range(full_masks.shape[2]):
                cv2.imwrite('test/full_mask_' + str(maskIndex) + '.png',
                            (full_masks[:, :, maskIndex] * 255).astype(np.uint8))
                continue
            pass
        return boxes, class_ids, scores, full_masks


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
	parse_image_meta() to parse the values back.

	image_id: An int ID of the image. Useful for debugging.
	image_shape: [height, width, channels]
	window: (y1, x1, y2, x2) in pixels. The area of the image where the real
			image is (excluding the padding)
	active_class_ids: List of class_ids available in the dataset from which
		the image came. Useful if training on images from multiple datasets
		where not all classes are present in all datasets.
	"""
    meta = np.array(
        [image_id] +  ## size=1
        list(image_shape) +  ## size=3
        list(window) +  ## size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  ## size=num_classes
    )
    return meta


## Two functions (for Numpy and TF) to parse image_meta tensors.
def parse_image_meta(meta):
    """Parses an image info Numpy array to its components.
	See compose_image_meta() for more details.
	"""
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]  ## (y1, x1, y2, x2) window of image in in pixels
    active_class_ids = meta[:, 8:]
    return image_id, image_shape, window, active_class_ids


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
	See compose_image_meta() for more details.

	meta: [batch, meta length] where meta length depends on NUM_CLASSES
	"""
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:8]
    active_class_ids = meta[:, 8:]
    return [image_id, image_shape, window, active_class_ids]


def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
	the mean pixel and converts it to float. Expects image
	colors in RGB order.
	"""
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)
