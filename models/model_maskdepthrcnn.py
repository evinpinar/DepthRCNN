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

import utils
import visualize
from tensorboardX import SummaryWriter
from nms.nms_wrapper import nms
from roialign.roi_align.crop_and_resize import CropAndResizeFunction
import cv2
from models.modules import *
from utils import *
from models.model import *
from nyu import NYUDataset


############################################################
##  Detection Target Layer
############################################################


def detection_target_layer_(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, MAX_GT_INSTANCES, height, width] of boolean type

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


def compute_gt_depths(gt_depths, gt_masks):
    """Extracts the masked depth values for each roi.

        gt_depth_map: [height, width].
            float32 tensor
        gt_masks: [rois, height, width]
            A float32 tensor of values between 0 and 1. Uses zero padding to fill array.

        returns:
            gt_masked_depths: [height, width, roi].
            A float32 tensor for each roi depth.
    """

    gt_masked_depths = np.zeros(gt_masks.shape)
    N = gt_masks.shape[2]
    for i in range(N):
        gt_masked_depths[:,:,i] = gt_depths * gt_masks[:,:,i]
    return gt_masked_depths


def compute_gt_normals(gt_normals, gt_masks):
    """Extracts the masked depth values for each roi.

        gt_normals: [3, height, width].
            float32 tensor
        gt_masks: [rois, height, width]
            A float32 tensor of values between 0 and 1. Uses zero padding to fill array.

        returns:
            gt_masked_depths: [height, width, 3, roi].
            A float32 tensor for each roi depth.
    """
    H, W, N = gt_masks.shape
    gt_masked_normals = np.zeros([H, W, 3, N])
    for i in range(N):
        gt_masked_normals[:, :, 0, i] = gt_normals[:, :, 0] * gt_masks[:, :, i]
        gt_masked_normals[:, :, 1, i] = gt_normals[:, :, 1] * gt_masks[:, :, i]
        gt_masked_normals[:, :, 2, i] = gt_normals[:, :, 2] * gt_masks[:, :, i]
    return gt_masked_normals


def shift_depth_target(target_mask, target_depth):
    '''
    Shifts the depth values of each roi. Substracts the minimum depth
    value for normalizing the depths (only in the masked region).

    :input target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
    :input target_depth: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
    :return: Same shaped target depth.
    '''

    N = target_mask.shape[0]
    for i in range(N):
        inds = np.where(target_mask[i] == 1)
        min_depth = torch.min(target_depth[i][inds])
        target_depth[i] -= min_depth

    return target_depth


def shift_depth(gt_masks, gt_depths):
    '''
    Shifts the depth values of each roi. Substracts the minimum depth
    value for normalizing the depths (only in the masked region).

    :input target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
    :input target_depth: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
    :return: Same shaped target depth.
    '''

    #print(gt_masks.shape, gt_depths.shape)
    N = gt_masks.shape[2]
    for i in range(N):
        inds = np.where(gt_masks[:,:,i] == 1)
        min_depth = np.min(gt_depths[:,:,i][inds])
        gt_depths[:,:,i] -= min_depth

    return gt_depths


def rescale_depth_target(target_depth, gt_boxes, mini_shape):
    '''
    Takes the target masked depths of each roi. Calculates delta, the
    number of pixels the object spans in original image. Then rescales
    the depth according to the ratio between original and mini shape.
    :param target_depth:
    :type target_depth: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    :param gt_boxes:
    :type gt_boxes: [TRAIN_ROIS_PER_IMAGE, height, width]
    :param mini_shape:
    :type mini_shape: [h, w]
    :return: rescaled depth
    :rtype: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
    '''
    N = target_depth.shape[0]
    for i in range(N):
        y1, x1, y2, x2 = gt_boxes[i][:4]
        delta = max(y2-y1, x2-x1) # Ideally they should be equal, aspect ratio = 1
        target_depth[i] = target_depth[i] * (delta / mini_shape[0])
    return target_depth

def detection_target_depth(proposals, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals, config):
    """ Subsamples proposals and generates target box refinment, class_ids,
        and masks for each.

        Inputs:
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                  coordinates.
        gt_masks: [batch, MAX_GT_INSTANCES, height, width] of boolean type
        gt_depths: [batch, height, width] of float32 type
        gt_normals: [batch, height, width, C=3] of float32 type

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        masks and depths.
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
    gt_masked_depths = gt_depths.squeeze(0)
    if config.PREDICT_NORMAL:
        gt_masked_normals = gt_normals.squeeze(0)
    # print("normals: ", gt_masked_normals.shape)
    #print("mask:", gt_masks.shape)
    #print("depth:", gt_masked_depths.shape)
    #gt_masked_depths = compute_gt_depths(gt_depths, gt_masks)

    no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
    if config.GPU_COUNT:
        gt_masked_depths = gt_masked_depths.cuda()
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
        roi_depths = gt_masked_depths[roi_gt_box_assignment.data]
        if config.PREDICT_NORMAL:
            roi_normals = gt_masked_normals[roi_gt_box_assignment.data]
        # print(" roi normals: ", roi_normals.shape)
        # print(" roi masks: ", roi_masks.shape)

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

        masks = Variable(
            CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes,
                                                                                 box_ids).data,
            requires_grad=False).squeeze(1)
        masks = torch.round(masks)

        depths = Variable(
            CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_depths.unsqueeze(1), boxes,
                                                                                 box_ids).data,
            requires_grad=False).squeeze(1)

        if config.PREDICT_NORMAL:
            normals = Variable(
                CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_normals, boxes,
                                                                                     box_ids).data,
                requires_grad=False)

        # print("after crop mask and normals ", masks.shape, normals.shape)
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
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                             requires_grad=False)
        zeros_normal = Variable(torch.zeros(negative_count, 3, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                                requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            zeros_normal = zeros_normal.cuda()

        # print("masks, normals, zeros before: ", masks.shape, normals.shape, zeros.shape)
        masks = torch.cat([masks, zeros], dim=0)
        depths = torch.cat([depths, zeros], dim=0)
        # print("then masks: ", masks.shape)
        if config.PREDICT_NORMAL:
            normals = torch.cat([normals, zeros_normal], dim=0)

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
        zeros_normal = Variable(torch.zeros(negative_count, 3, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                                requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            zeros_normal = zeros_normal.cuda()
        masks = zeros
        depths = zeros
        normals = zeros_normal

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        depths = Variable(torch.FloatTensor(), requires_grad=False)
        normals = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
            depths = depths.cuda()
            normals = normals.cuda()
            pass

    if not config.PREDICT_NORMAL:
        normals = Variable(torch.FloatTensor(), requires_grad=False)
        normals = normals.cuda()

    return rois, roi_gt_class_ids, deltas, masks, depths, normals


def detection_target_depth2(proposals, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals, config):
    """ Subsamples proposals and generates target box refinment, class_ids,
        and masks for each.

        Inputs:
        proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
                   be zero padded if there are not enough proposals.
        gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
        gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                  coordinates.
        gt_masks: [batch, MAX_GT_INSTANCES, height, width] of boolean type
        gt_depths: [batch, height, width] of float32 type
        gt_normals: [batch, height, width, C=3] of float32 type

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
        masks and depths.
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
    gt_masked_depths = gt_depths.squeeze(0)
    if config.PREDICT_NORMAL:
        gt_masked_normals = gt_normals.squeeze(0)
    # print("normals: ", gt_masked_normals.shape)
    #print("mask:", gt_masks.shape)
    #print("depth:", gt_masked_depths.shape)
    #gt_masked_depths = compute_gt_depths(gt_depths, gt_masks)

    no_crowd_bool = Variable(torch.ByteTensor(proposals.size()[0] * [True]), requires_grad=False)
    if config.GPU_COUNT:
        gt_masked_depths = gt_masked_depths.cuda()
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
        roi_depths = gt_masked_depths[roi_gt_box_assignment.data]
        if config.PREDICT_NORMAL:
            roi_normals = gt_masked_normals[roi_gt_box_assignment.data]
        # print(" roi normals: ", roi_normals.shape)
        # print(" roi masks: ", roi_masks.shape)

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

        masks = Variable(
            CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes,
                                                                                 box_ids).data,
            requires_grad=False).squeeze(1)
        masks = torch.round(masks)

        depths = Variable(
            CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_depths.unsqueeze(1), boxes,
                                                                                 box_ids).data,
            requires_grad=False).squeeze(1)

        if config.PREDICT_NORMAL:
            normals = Variable(
                CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_normals, boxes,
                                                                                     box_ids).data,
                requires_grad=False)

        # print("after crop mask and normals ", masks.shape, normals.shape)
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
        zeros = Variable(torch.zeros(negative_count, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                             requires_grad=False)
        zeros_normal = Variable(torch.zeros(negative_count, 3, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                                requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            zeros_normal = zeros_normal.cuda()

        # print("masks, normals, zeros before: ", masks.shape, normals.shape, zeros.shape)
        masks = torch.cat([masks, zeros], dim=0)
        depths = torch.cat([depths, zeros], dim=0)
        # print("then masks: ", masks.shape)
        if config.PREDICT_NORMAL:
            normals = torch.cat([normals, zeros_normal], dim=0)

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
        zeros_normal = Variable(torch.zeros(negative_count, 3, config.MASK_SHAPE[0], config.MASK_SHAPE[1]),
                                requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
            zeros_normal = zeros_normal.cuda()
        masks = zeros
        depths = zeros
        normals = zeros_normal

        zeros = Variable(torch.zeros(negative_count, config.NUM_PARAMETERS), requires_grad=False)
        if config.GPU_COUNT:
            zeros = zeros.cuda()
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        depths = Variable(torch.FloatTensor(), requires_grad=False)
        normals = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()
            depths = depths.cuda()
            normals = normals.cuda()
            pass

    if not config.PREDICT_NORMAL:
        normals = Variable(torch.FloatTensor(), requires_grad=False)
        normals = normals.cuda()

    return rois, roi_gt_class_ids, deltas, masks, depths, normals


def data_generator2(dataset, config, shuffle=True, augment=False, augmentation=None,
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
    - gt_depth: [batch, height, width]. The height and width
              are those of the image.
    - gt_normals: [batch, 3, height, width]. The height and width
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

            # print(gt_masks.shape)
            # print(gt_depth.shape)
            gt_depths = compute_gt_depths(gt_depth, gt_masks) # Mask the depths
            if config.PREDICT_NORMAL:
                gt_normals = compute_gt_normals(gt_normal, gt_masks) # Mask the normals

            # Resize masks to smaller size to reduce memory usage
            if config.USE_MINI_MASK:
                gt_masks = utils.minimize_mask(gt_boxes, gt_masks, config.MINI_MASK_SHAPE)
                gt_depths = utils.minimize_depth(gt_boxes, gt_depths, config.MINI_MASK_SHAPE)

                if config.PREDICT_NORMAL:
                    gt_normals = utils.minimize_normal(gt_boxes, gt_normals, config.MINI_MASK_SHAPE)


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
                batch_gt_depths = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=np.float32)
                batch_gt_normals = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1], 3,
                     config.MAX_GT_INSTANCES), dtype=np.float32)
                #batch_depth = np.zeros((batch_size,) + gt_depth.shape, dtype=np.float32)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
                gt_depths = gt_depths[:, :, ids]
                if config.PREDICT_NORMAL:
                    gt_normals = gt_normals[:, :, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_depths[b, :, :, :gt_masks.shape[-1]] = gt_depths.astype(np.float32)
            if config.PREDICT_NORMAL:
                batch_gt_normals[b, :, :, :, :gt_masks.shape[-1]] = gt_normals.astype(np.float32)
            else:
                batch_gt_normals = np.zeros([1,1,1,1,1])
            #gt_masked_depths[b] = gt_depth.astype(np.float32)
            b += 1

            # Batch full?
            if b >= batch_size:

                #depth = np.zeros(1)

                yield [torch.from_numpy(batch_images.transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_image_meta), torch.from_numpy(batch_rpn_match),
                       torch.from_numpy(batch_rpn_bbox.astype(np.float32)), torch.from_numpy(batch_gt_class_ids),
                       torch.from_numpy(batch_gt_boxes.astype(np.float32)),
                       torch.from_numpy(batch_gt_masks.astype(np.float32).transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_gt_depths.astype(np.float32).transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_gt_normals.astype(np.float32).transpose(0, 4, 3, 1, 2))]
                       #torch.from_numpy(batch_depth.astype(np.float32))]

                #inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                #          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                #outputs = []


                #yield inputs, outputs

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

## Adds padding to keep aspect ratio.
def data_generator3(dataset, config, shuffle=True, augment=False, augmentation=None,
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
    - gt_depth: [batch, height, width]. The height and width
              are those of the image.
    - gt_normals: [batch, 3, height, width]. The height and width
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

            # print(gt_masks.shape)
            # print(gt_depth.shape)
            gt_depths = compute_gt_depths(gt_depth, gt_masks) # Mask the depths
            if config.PREDICT_NORMAL:
                gt_normals = compute_gt_normals(gt_normal, gt_masks) # Mask the normals

            # Resize masks to smaller size to reduce memory usage
            if config.USE_MINI_MASK:
                gt_masks = utils.minimize_mask_square(gt_boxes, gt_masks, config.MINI_MASK_SHAPE)
                gt_depths = utils.minimize_depth_square(gt_boxes, gt_depths, config.MINI_MASK_SHAPE)

                if config.PREDICT_NORMAL:
                    gt_normals = utils.minimize_normal(gt_boxes, gt_normals, config.MINI_MASK_SHAPE)


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
                batch_gt_depths = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=np.float32)
                batch_gt_normals = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1], 3,
                     config.MAX_GT_INSTANCES), dtype=np.float32)
                #batch_depth = np.zeros((batch_size,) + gt_depth.shape, dtype=np.float32)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
                gt_depths = gt_depths[:, :, ids]
                if config.PREDICT_NORMAL:
                    gt_normals = gt_normals[:, :, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_depths[b, :, :, :gt_masks.shape[-1]] = gt_depths.astype(np.float32)
            if config.PREDICT_NORMAL:
                batch_gt_normals[b, :, :, :, :gt_masks.shape[-1]] = gt_normals.astype(np.float32)
            else:
                batch_gt_normals = np.zeros([1,1,1,1,1])
            #gt_masked_depths[b] = gt_depth.astype(np.float32)
            b += 1

            # Batch full?
            if b >= batch_size:

                #depth = np.zeros(1)

                yield [torch.from_numpy(batch_images.transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_image_meta), torch.from_numpy(batch_rpn_match),
                       torch.from_numpy(batch_rpn_bbox.astype(np.float32)), torch.from_numpy(batch_gt_class_ids),
                       torch.from_numpy(batch_gt_boxes.astype(np.float32)),
                       torch.from_numpy(batch_gt_masks.astype(np.float32).transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_gt_depths.astype(np.float32).transpose(0, 3, 1, 2)),
                       torch.from_numpy(batch_gt_normals.astype(np.float32).transpose(0, 4, 3, 1, 2))]
                       #torch.from_numpy(batch_depth.astype(np.float32))]

                #inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                #          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                #outputs = []


                #yield inputs, outputs

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
#  Depth Head for MaskRCNN
############################################################

class DepthMask(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(DepthMask, self).__init__()
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
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, pool_features=True):
        if pool_features:
            roi_features = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
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
        # print(" roi depth before last conv: ", x.shape)
        x = self.conv5(x)
        # print(" roi depth after last conv: ", x.shape)
        #x = self.sigmoid(x)

        return x, roi_features


class Depth(nn.Module):
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
        x = self.depth_pred(x)

        if self.crop:
            x = torch.nn.functional.interpolate(x, size=(480, 640), mode='bilinear')
            zeros = torch.zeros((len(x), self.num_output_channels, 80, 640)).cuda()
            x = torch.cat([zeros, x, zeros], dim=2)
        else:
            x = torch.nn.functional.interpolate(x, size=(640, 640), mode='bilinear')
            pass
        return x


class NormalMask(nn.Module):
    def __init__(self, config, depth, pool_size, image_shape, num_classes):
        super(NormalMask, self).__init__()
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

        # Multiply output by 3 because 3 dim vector outputs for each class.
        self.conv5 = nn.Conv2d(256, num_classes*3, kernel_size=1, stride=1)

        self.outputs = torch.zeros([1, num_classes, 3, image_shape[0], image_shape[1]])
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, rois, pool_features=True):
        if pool_features:
            roi_features = pyramid_roi_align([rois] + x, self.pool_size, self.image_shape)
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

        return x, roi_features


############################################################
#  Loss Functions
############################################################

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

def compute_mrcnn_depth_loss(config, target_depths, target_class_ids, pred_depths):
    """Depth loss for the masks head.

        target_depths: [batch, num_rois, height, width].
            A float32 tensor of values between 0 and 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values between 0 and 1.
        """
    if (target_class_ids > 0).sum() > 0:

        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_depths[indices[:, 0].data, :, :]
        y_pred = pred_depths[indices[:, 0].data, indices[:, 1].data, :, :]

        # Mask_loss: loss = F.binary_cross_entropy(y_pred, y_true)

        # proportion of 480-640 -> 21-28
        #loss = l1LossMask(y_pred[:, 3:25], y_true[:, 3:25],
        #				  (y_true[:, 3:25] > 1e-4).float())

        if config.DEPTH_LOSS == 'L1':
            loss = compute_depth_loss_L1(y_true, y_pred, config.DEPTH_THRESHOLD)
        if config.DEPTH_LOSS == 'L2':
            loss = compute_depth_loss_L2(y_true, y_pred, config.DEPTH_THRESHOLD)
        if config.DEPTH_LOSS == 'BERHU':
            loss = compute_depth_loss_berHu(y_true, y_pred, config.DEPTH_THRESHOLD)
    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss


def compute_mrcnn_plane_loss(config, target_normal, target_class_ids, pred_normal):
    """Surface normal loss for the masks head.

        target_planes: [batch, num_rois, 3, height, width].
            A float32 tensor of values between 0 and 1. Uses zero padding to fill array.
        target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
        pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                    with values between 0 and 1.
        """
    if (target_class_ids > 0).sum() > 0:

        ## Only positive ROIs contribute to the loss. And only
        ## the class specific mask of each ROI.
        positive_ix = torch.nonzero(target_class_ids > 0)[:, 0]
        positive_class_ids = target_class_ids[positive_ix.data].long()
        indices = torch.stack((positive_ix, positive_class_ids), dim=1)

        ## Gather the masks (predicted and true) that contribute to loss
        y_true = target_normal[indices[:, 0].data, :, :, :]
        y_pred = pred_normal[indices[:, 0].data, indices[:, 1].data, :, :, :]


        loss = l2LossMask(y_pred, y_true, (y_true > 0).float())

    else:
        loss = Variable(torch.FloatTensor([0]), requires_grad=False)
        if target_class_ids.is_cuda:
            loss = loss.cuda()

    return loss

## Loss function with maskrcnn_depth loss
def compute_losses_(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                    target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_depth, pred_depth, target_normal, pred_normal):
    rpn_class_loss = compute_rpn_class_loss(rpn_match, rpn_class_logits)
    rpn_bbox_loss = compute_rpn_bbox_loss(rpn_bbox, rpn_match, rpn_pred_bbox)
    mrcnn_class_loss = compute_mrcnn_class_loss(target_class_ids, mrcnn_class_logits)
    mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_deltas, target_class_ids, mrcnn_bbox)
    mrcnn_mask_loss = compute_mrcnn_mask_loss(config, target_mask, target_class_ids, mrcnn_mask)

    depth_loss = torch.tensor([0], dtype=torch.float32)
    if config.PREDICT_DEPTH:
        depth_loss = compute_mrcnn_depth_loss(config, target_depth, target_class_ids, pred_depth)

    normal_loss = torch.tensor([0], dtype=torch.float32)
    if config.PREDICT_NORMAL:
        normal_loss = compute_mrcnn_plane_loss(config, target_normal, target_class_ids, pred_normal)
    return [rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, normal_loss]


############################################################
#  MaskDepthRCNN Class for Depth (from ROIs)
############################################################

class MaskDepthRCNN(nn.Module):
    """Encapsulates the Mask RCNN model functionality for
    bounding box prediction, mask prediction, object classification,
    and depth prediction for each object.
    """

    def __init__(self, config, model_dir='checkpoints'):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskDepthRCNN, self).__init__()
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
        self.anchors = Variable(torch.from_numpy(utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                                                config.RPN_ANCHOR_RATIOS,
                                                                                config.BACKBONE_SHAPES,
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

        ## FPN Depth
        self.depthmask = DepthMask(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

        ## FPN Normal
        self.normalmask = NormalMask(config, 256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)

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
        self.checkpoint_path = os.path.join(self.log_dir, "mask_depth_rcnn_{}_*epoch*.pth".format(
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
        checkpoints = filter(lambda f: f.startswith("mask_depth_rcnn"), checkpoints)
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
                    print('load only base model')
                    try:
                        state_dict = {k: v for k, v in state_dict.items() if
                                      'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k}
                        state = self.state_dict()
                        state.update(state_dict)
                        self.load_state_dict(state)
                    except:
                        print('change input dimension')
                        state_dict = {k: v for k, v in state_dict.items() if
                                      'classifier.linear_class' not in k and 'classifier.linear_bbox' not in k and 'mask.conv5' not in k and 'fpn.C1.0' not in k and 'classifier.conv1' not in k}
                        state = self.state_dict()
                        state.update(state_dict)
                        self.load_state_dict(state)
                        pass
                    pass
        else:
            print("Weight file not found ...")
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
        detections, mrcnn_mask, mrcnn_depth, mrcnn_normals = self.predict([molded_images, image_metas], mode='inference')

        if len(detections[0]) == 0:
            return [{'rois': [], 'class_ids': [], 'scores': [], 'masks': [], 'parameters': []}]

        ## Convert to numpy
        detections = detections.data.cpu().numpy()
        mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()
        if self.config.PREDICT_DEPTH:
            mrcnn_depth = mrcnn_depth.permute(0, 1, 3, 4, 2).data.cpu().numpy()

        ## Process detections
        results = []
        #final_normals = mrcnn_normals
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks, final_depths, final_normals = \
                self.unmold_detections(self.config, detections[i], mrcnn_mask[i],
                                       mrcnn_depth[i], mrcnn_normals[i], image.shape, windows[i])

            print(" mask shape: ", mrcnn_mask.shape)
            print(" unmolded mask shape: ", final_masks.shape)

            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "depths": final_depths,
                "normals": final_normals,
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
                return [[]], [[]], [[]]
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

            ## Create depth for detections
            if self.config.PREDICT_DEPTH:
                mrcnn_depth, _ = self.depthmask(mrcnn_feature_maps, detection_boxes)
            else:
                mrcnn_depth = Variable(torch.FloatTensor([[1, 0]])).cuda()

            ## Create depth for detections
            if self.config.PREDICT_NORMAL:
                mrcnn_normal, _ = self.normalmask(mrcnn_feature_maps, detection_boxes)
                # print("normal shape: ", mrcnn_normal.shape)
                rois, out_channels, H, W = mrcnn_normal.shape
                num_classes = int(out_channels/3)
                mrcnn_normal = torch.reshape(mrcnn_normal, (rois, num_classes, 3, H, W))
            else:
                mrcnn_normal = Variable(torch.FloatTensor([[1, 0]])).cuda()

            ## Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            mrcnn_depth = mrcnn_depth.unsqueeze(0)
            mrcnn_normal = mrcnn_normal.unsqueeze(0)
            return [detections, mrcnn_mask, mrcnn_depth, mrcnn_normal]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_depths = input[5]
            gt_normals = input[6]

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
            rois, target_class_ids, target_deltas, target_mask, target_depth, target_normals = \
                detection_target_depth(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals, self.config)

            if self.config.PREDICT_DEPTH:
                target_depth = shift_depth_target(target_mask, target_depth)
                #target_depth = rescale_depth_target(target_depth, gt_boxes, config.MINI_MASK_SHAPE)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_depth = Variable(torch.FloatTensor())
                mrcnn_normal = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_depth = mrcnn_depth.cuda()
                    mrcnn_normal = mrcnn_normal.cuda()
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
                mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, rois)

                ## Create depth for detections
                if self.config.PREDICT_DEPTH:
                    mrcnn_depth, _ = self.depthmask(mrcnn_feature_maps, rois)
                else:
                    mrcnn_depth = Variable(torch.FloatTensor()).cuda()

                ## Create depth for detections
                if self.config.PREDICT_NORMAL:
                    mrcnn_normal, _ = self.normalmask(mrcnn_feature_maps, rois)
                    rois, out_channels, H, W = mrcnn_normal.shape
                    num_classes = int(out_channels / 3)
                    mrcnn_normal = torch.reshape(mrcnn_normal, (rois, num_classes, 3, H, W))
                else:
                    mrcnn_normal = Variable(torch.FloatTensor()).cuda()

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, rois, target_depth, mrcnn_depth, target_normals, mrcnn_normal]

    ## Scaled roi prediction instead of shifting
    def predict2(self, input, mode, use_nms=1, use_refinement=False, return_feature_map=False):
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
                return [[]], [[]], [[]]
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

            ## Create depth for detections
            if self.config.PREDICT_DEPTH:
                mrcnn_depth, _ = self.depthmask(mrcnn_feature_maps, detection_boxes)
            else:
                mrcnn_depth = Variable(torch.FloatTensor([[1, 0]])).cuda()

            ## Create depth for detections
            if self.config.PREDICT_NORMAL:
                mrcnn_normal, _ = self.normalmask(mrcnn_feature_maps, detection_boxes)
                # print("normal shape: ", mrcnn_normal.shape)
                rois, out_channels, H, W = mrcnn_normal.shape
                num_classes = int(out_channels/3)
                mrcnn_normal = torch.reshape(mrcnn_normal, (rois, num_classes, 3, H, W))
            else:
                mrcnn_normal = Variable(torch.FloatTensor([[1, 0]])).cuda()

            ## Add back batch dimension
            detections = detections.unsqueeze(0)
            mrcnn_mask = mrcnn_mask.unsqueeze(0)
            mrcnn_depth = mrcnn_depth.unsqueeze(0)
            mrcnn_normal = mrcnn_normal.unsqueeze(0)
            return [detections, mrcnn_mask, mrcnn_depth, mrcnn_normal]

        elif mode == 'training':

            gt_class_ids = input[2]
            gt_boxes = input[3]
            gt_masks = input[4]
            gt_depths = input[5]
            gt_normals = input[6]

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
            rois, target_class_ids, target_deltas, target_mask, target_depth, target_normals = \
                detection_target_depth2(rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals, self.config)

            if self.config.PREDICT_DEPTH:
                #target_depth = shift_depth_target(target_mask, target_depth)
                target_depth = rescale_depth_target(target_depth, gt_boxes[0], self.config.MINI_MASK_SHAPE)

            if len(rois) == 0:
                mrcnn_class_logits = Variable(torch.FloatTensor())
                mrcnn_class = Variable(torch.IntTensor())
                mrcnn_bbox = Variable(torch.FloatTensor())
                mrcnn_mask = Variable(torch.FloatTensor())
                mrcnn_depth = Variable(torch.FloatTensor())
                mrcnn_normal = Variable(torch.FloatTensor())
                if self.config.GPU_COUNT:
                    mrcnn_class_logits = mrcnn_class_logits.cuda()
                    mrcnn_class = mrcnn_class.cuda()
                    mrcnn_bbox = mrcnn_bbox.cuda()
                    mrcnn_mask = mrcnn_mask.cuda()
                    mrcnn_depth = mrcnn_depth.cuda()
                    mrcnn_normal = mrcnn_normal.cuda()
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
                mrcnn_mask, roi_features = self.mask(mrcnn_feature_maps, rois)

                ## Create depth for detections
                if self.config.PREDICT_DEPTH:
                    mrcnn_depth, _ = self.depthmask(mrcnn_feature_maps, rois)
                else:
                    mrcnn_depth = Variable(torch.FloatTensor()).cuda()

                ## Create depth for detections
                if self.config.PREDICT_NORMAL:
                    mrcnn_normal, _ = self.normalmask(mrcnn_feature_maps, rois)
                    rois, out_channels, H, W = mrcnn_normal.shape
                    num_classes = int(out_channels / 3)
                    mrcnn_normal = torch.reshape(mrcnn_normal, (rois, num_classes, 3, H, W))
                else:
                    mrcnn_normal = Variable(torch.FloatTensor()).cuda()

            return [rpn_class_logits, rpn_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox,
                    target_mask, mrcnn_mask, rois, target_depth, mrcnn_depth, target_normals, mrcnn_normal]

    def train_model2(self, train_dataset, val_dataset, learning_rate, epochs, layers, depth_weight=1, augmentation=None):
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
              heads: The RPN, classifier and mask heads of the network
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
        train_generator = data_generator2(train_dataset, self.config, shuffle=True, augment=False, batch_size=1, augmentation=augmentation)
        val_generator = data_generator2(val_dataset, self.config, shuffle=True, augment=False, batch_size=1, augmentation=augmentation)

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
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask,\
            loss_depth, loss_normal = self.train_epoch(
                train_generator, optimizer, self.config.STEPS_PER_EPOCH, depth_weight)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, \
            val_loss_mrcnn_mask, val_loss_depth, val_loss_normal = self.valid_epoch(
                val_generator, self.config.VALIDATION_STEPS, depth_weight)

            # Statistics
            self.loss_history.append(
                [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask,
                 loss_depth, loss_normal])
            self.val_loss_history.append(
                [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
                 val_loss_mrcnn_mask, val_loss_depth, val_loss_normal])
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
            writer.add_scalar('Train/Normal', loss_normal, epoch)

            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/RPN_Class', val_loss_rpn_class, epoch)
            writer.add_scalar('Val/RPN_BBOX', val_loss_rpn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Class', val_loss_mrcnn_class, epoch)
            writer.add_scalar('Val/MRCNN_BBOX', val_loss_mrcnn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Mask', val_loss_mrcnn_mask, epoch)
            writer.add_scalar('Val/Depth', val_loss_depth, epoch)
            writer.add_scalar('Val/Normal', val_loss_normal, epoch)

            # Save model
            if epoch % 5 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_model(self, train_dataset, val_dataset, learning_rate, epochs, layers, depth_weight=1):
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
        trainables_only_bn = [param for name, param in self.named_parameters() if
                              param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)

        for epoch in range(self.epoch + 1, epochs + 1):
            log("Epoch {}/{}.".format(epoch, epochs))

            # Training
            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask, \
                loss_depth, loss_normal = self.train_epoch(
                train_generator, optimizer, self.config.STEPS_PER_EPOCH, depth_weight)

            # Validation
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, \
                val_loss_mrcnn_mask, val_loss_depth, val_loss_normal = self.valid_epoch(
                val_generator, self.config.VALIDATION_STEPS, depth_weight)

            # Statistics
            self.loss_history.append(
                [loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask,
                 loss_depth, loss_normal])
            self.val_loss_history.append(
                [val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox,
                 val_loss_mrcnn_mask, val_loss_depth, val_loss_normal])
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
            writer.add_scalar('Train/Normal', loss_normal, epoch)

            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/RPN_Class', val_loss_rpn_class, epoch)
            writer.add_scalar('Val/RPN_BBOX', val_loss_rpn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Class', val_loss_mrcnn_class, epoch)
            writer.add_scalar('Val/MRCNN_BBOX', val_loss_mrcnn_bbox, epoch)
            writer.add_scalar('Val/MRCNN_Mask', val_loss_mrcnn_mask, epoch)
            writer.add_scalar('Val/Depth', val_loss_depth, epoch)
            writer.add_scalar('Val/Normal', val_loss_normal, epoch)

            # Save model
            if epoch % 25 == 0:
                torch.save(self.state_dict(), self.checkpoint_path.format(epoch))

        self.epoch = epochs

    def train_epoch(self, datagenerator, optimizer, steps, depth_weight=1, normal_weight=0.001):
        batch_count = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_depth_sum = 0
        loss_normal_sum = 0
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
            gt_normals = inputs[8]

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
            gt_normals = Variable(gt_normals)

            # To GPU
            if self.config.GPU_COUNT:
                images = images.cuda()
                rpn_match = rpn_match.cuda()
                rpn_bbox = rpn_bbox.cuda()
                gt_class_ids = gt_class_ids.cuda()
                gt_boxes = gt_boxes.cuda()
                gt_masks = gt_masks.cuda()
                gt_depths = gt_depths.cuda()
                gt_normals = gt_normals.cuda()


            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, rois, target_depths, mrcnn_depths, target_normal, mrcnn_normal = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals], mode='training')

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, normal_loss = compute_losses_(
                self.config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits,
                target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_depths, mrcnn_depths, target_normal, mrcnn_normal)

            if self.config.GPU_COUNT:
                depth_loss = depth_loss.cuda()
                normal_loss = normal_loss.cuda()

            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss \
                   + depth_weight*depth_loss + normal_weight*normal_loss

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
                                 suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - depth_loss: {:.5f} - normal_loss: {:.5f}".format(
                                     loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                     rpn_bbox_loss.data.cpu().item(),
                                     mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                     mrcnn_mask_loss.data.cpu().item(), depth_loss.data.cpu().item(),
                                     normal_loss.data.cpu().item()), length=10)

            # Statistics
            loss_sum += loss.data.cpu().item() / steps
            loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
            loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item() / steps
            loss_depth_sum += depth_loss.data.cpu().item() / steps
            loss_normal_sum += normal_loss.data.cpu().item() / steps

            # Break after 'steps' steps
            if step == steps - 1:
                break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, \
               loss_mrcnn_mask_sum, loss_depth_sum, loss_normal_sum

    def valid_epoch(self, datagenerator, steps, depth_weight=1, normal_weight=0.001):

        step = 0
        loss_sum = 0
        loss_rpn_class_sum = 0
        loss_rpn_bbox_sum = 0
        loss_mrcnn_class_sum = 0
        loss_mrcnn_bbox_sum = 0
        loss_mrcnn_mask_sum = 0
        loss_depth_sum = 0
        loss_normal_sum = 0

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
                gt_normals = inputs[8]

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
                gt_normals = Variable(gt_normals)

                # To GPU
                if self.config.GPU_COUNT:
                    images = images.cuda()
                    rpn_match = rpn_match.cuda()
                    rpn_bbox = rpn_bbox.cuda()
                    gt_class_ids = gt_class_ids.cuda()
                    gt_boxes = gt_boxes.cuda()
                    gt_masks = gt_masks.cuda()
                    gt_depths = gt_depths.cuda()
                    gt_normals = gt_normals.cuda()

                # Run object detection
                rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, rois, target_depths, mrcnn_depths, target_normals, mrcnn_normals = \
                    self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks, gt_depths, gt_normals], mode='training')

                if not target_class_ids.size():
                    continue

                # Compute losses
                rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, depth_loss, normal_loss = compute_losses_(
                    self.config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids,
                    mrcnn_class_logits,
                    target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, target_depths, mrcnn_depths,
                    target_normals, mrcnn_normals)
                if self.config.GPU_COUNT:
                    depth_loss = depth_loss.cuda()
                    normal_loss = normal_loss.cuda()

                loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss \
                       + depth_weight*depth_loss + normal_weight*normal_loss

                # Progress
                if step % 100 == 0:
                    printProgressBar(step + 1, steps, prefix="\t{}/{}".format(step + 1, steps),
                                     suffix="Complete - loss: {:.5f} - rpn_class_loss: {:.5f} - rpn_bbox_loss: {:.5f} - mrcnn_class_loss: {:.5f} - mrcnn_bbox_loss: {:.5f} - mrcnn_mask_loss: {:.5f} - depth_loss: {:.5f} - normal_loss: {:.5f}".format(
                                         loss.data.cpu().item(), rpn_class_loss.data.cpu().item(),
                                         rpn_bbox_loss.data.cpu().item(),
                                         mrcnn_class_loss.data.cpu().item(), mrcnn_bbox_loss.data.cpu().item(),
                                         mrcnn_mask_loss.data.cpu().item(), depth_loss.data.cpu().item(),
                                         normal_loss.data.cpu().item()), length=10)

                # Statistics
                loss_sum += loss.data.cpu().item() / steps
                loss_rpn_class_sum += rpn_class_loss.data.cpu().item() / steps
                loss_rpn_bbox_sum += rpn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_class_sum += mrcnn_class_loss.data.cpu().item() / steps
                loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.cpu().item() / steps
                loss_mrcnn_mask_sum += mrcnn_mask_loss.data.cpu().item() / steps
                loss_depth_sum += depth_loss.data.cpu().item() / steps
                loss_normal_sum += normal_loss.data.cpu().item() / steps

                # Break after 'steps' steps
                if step == steps - 1:
                    break
                step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, \
               loss_mrcnn_mask_sum, loss_depth_sum, loss_normal_sum

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
                min_dim=config.IMAGE_MAX_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                mode=config.IMAGE_RESIZE_MODE)
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

    def unmold_detections(self, config, detections, mrcnn_mask, mrcnn_depth, mrcnn_normal, image_shape, window, debug=False):
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
            if config.PREDICT_DEPTH:
                depths = mrcnn_mask[np.arange(N), :, :, 0]
        else:
            masks = mrcnn_mask[np.arange(N), :, :, class_ids]
            if config.PREDICT_DEPTH:
                depths = mrcnn_depth[np.arange(N), :, :, class_ids]
            if config.PREDICT_NORMAL:
                #print("unmolding mrcnn normals: ", mrcnn_normal.shape)
                normals = mrcnn_normal[np.arange(N), class_ids, :, :, :]
                #print("unmolded normals ",normals.shape)
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
            print(masks.shape, boxes.shape)
            for maskIndex, mask in enumerate(masks):
                print(maskIndex, boxes[maskIndex].astype(np.int32))
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
            if config.PREDICT_DEPTH:
                depths = np.delete(depths, exclude_ix, axis=0)
            if config.PREDICT_NORMAL:
                normals = np.delete(normals, exclude_ix, axis=0)
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
            print(full_masks.shape)
            for maskIndex in range(full_masks.shape[2]):
                cv2.imwrite('test/full_mask_' + str(maskIndex) + '.png',
                            (full_masks[:, :, maskIndex] * 255).astype(np.uint8))
                continue
            pass

        full_depths = []
        if config.PREDICT_DEPTH:
            for i in range(N):
                ## Convert neural network mask to full size mask
                full_depth = utils.unmold_depth(depths[i], boxes[i], image_shape)
                full_depths.append(full_depth)
            full_depths = np.stack(full_depths, axis=-1) \
                if full_depths else np.empty((0,) + depths.shape[1:3])

        full_normals = []
        if config.PREDICT_NORMAL:
            full_normals = normals


        return boxes, class_ids, scores, full_masks, full_depths, full_normals



