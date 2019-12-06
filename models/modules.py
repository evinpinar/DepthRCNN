"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from torch import nn
import sys

def unmoldDetections(config, camera, detections, detection_masks, depth_np, unmold_masks=True, debug=False):
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
    if config.GLOBAL_MASK:
        masks = detection_masks[torch.arange(len(detection_masks)).cuda().long(), 0, :, :]
    else:
        masks = detection_masks[torch.arange(len(detection_masks)).cuda().long(), detections[:, 4].long(), :, :]
        pass

    final_masks = []
    for detectionIndex in range(len(detections)):
        box = detections[detectionIndex][:4].long()
        if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
            continue
            
        mask = masks[detectionIndex]
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.upsample(mask, size=(box[2] - box[0], box[3] - box[1]), mode='bilinear')
        mask = mask.squeeze(0).squeeze(0)

        final_mask = torch.zeros(config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM).cuda()
        final_mask[box[0]:box[2], box[1]:box[3]] = mask
        final_masks.append(final_mask)
        continue
    final_masks = torch.stack(final_masks, dim=0)
    
    if config.NUM_PARAMETER_CHANNELS > 0:
        ## We could potentially predict depth and/or normals for each instance (not being used)
        parameters_array = detection_masks[torch.arange(len(detection_masks)).cuda().long(), -config.NUM_PARAMETER_CHANNELS:, :, :]
        final_parameters_array = []
        for detectionIndex in range(len(detections)):
            box = detections[detectionIndex][:4].long()
            if (box[2] - box[0]) * (box[3] - box[1]) <= 0:
                continue
            parameters = F.upsample(parameters_array[detectionIndex].unsqueeze(0), size=(box[2] - box[0], box[3] - box[1]), mode='bilinear').squeeze(0)
            final_parameters = torch.zeros(config.NUM_PARAMETER_CHANNELS, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM).cuda()
            final_parameters[:, box[0]:box[2], box[1]:box[3]] = parameters
            final_parameters_array.append(final_parameters)
            continue
        final_parameters = torch.stack(final_parameters_array, dim=0)        
        final_masks = torch.cat([final_masks.unsqueeze(1), final_parameters], dim=1)
        pass

    masks = final_masks

    if 'normal' in config.ANCHOR_TYPE:
        ## Compute offset based normal prediction and depthmap prediction
        ranges = config.getRanges(camera).transpose(1, 2).transpose(0, 1)
        zeros = torch.zeros(3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()        
        ranges = torch.cat([zeros, ranges, zeros], dim=1)
        
        if config.NUM_PARAMETER_CHANNELS == 4:
            ## If we predict depthmap and normal map for each instance, we compute normals again (not used)
            masks_cropped = masks[:, 0:1, 80:560]
            mask_sum = masks_cropped.sum(-1).sum(-1)
            plane_normals = (masks[:, 2:5, 80:560] * masks_cropped).sum(-1).sum(-1) / mask_sum
            plane_normals = plane_normals / torch.clamp(torch.norm(plane_normals, dim=-1, keepdim=True), min=1e-4)
            XYZ_np_cropped = (ranges * masks[:, 1:2])[:, :, 80:560]
            offsets = ((plane_normals.view(-1, 3, 1, 1) * XYZ_np_cropped).sum(1, keepdim=True) * masks_cropped).sum(-1).sum(-1) / mask_sum
            plane_parameters = plane_normals * offsets.view((-1, 1))
            masks = masks[:, 0]            
        else:
            if config.NUM_PARAMETER_CHANNELS > 0:
                ## If we predict depthmap independently for each instance, we use the individual depthmap instead of the global depth map (not used)                
                if config.OCCLUSION:
                    XYZ_np = ranges * depth_np                
                    XYZ_np_cropped = XYZ_np[:, 80:560]
                    masks_cropped = masks[:, 1, 80:560]                    
                    masks = masks[:, 0]
                else:
                    XYZ_np_cropped = (ranges * masks[:, 1:2])[:, :, 80:560]
                    masks = masks[:, 0]
                    masks_cropped = masks[:, 80:560]
                    pass
            else:
                ## We use the global depthmap prediction to compute plane offsets
                XYZ_np = ranges * depth_np                
                XYZ_np_cropped = XYZ_np[:, 80:560]
                masks_cropped = masks[:, 80:560]                            
                pass

            if config.FITTING_TYPE % 2 == 1:
                ## We fit all plane parameters using depthmap prediction (not used)
                A = masks_cropped.unsqueeze(1) * XYZ_np_cropped
                b = masks_cropped
                Ab = (A * b.unsqueeze(1)).sum(-1).sum(-1)
                AA = (A.unsqueeze(2) * A.unsqueeze(1)).sum(-1).sum(-1)
                plane_parameters = torch.stack([torch.matmul(torch.inverse(AA[planeIndex]), Ab[planeIndex]) for planeIndex in range(len(AA))], dim=0)
                plane_offsets = torch.norm(plane_parameters, dim=-1, keepdim=True)
                plane_parameters = plane_parameters / torch.clamp(torch.pow(plane_offsets, 2), 1e-4)                
            else:
                ## We compute only plane offset using depthmap prediction                
                plane_parameters = detections[:, 6:9]            
                plane_normals = plane_parameters / torch.clamp(torch.norm(plane_parameters, dim=-1, keepdim=True), 1e-4)
                offsets = ((plane_normals.view(-1, 3, 1, 1) * XYZ_np_cropped).sum(1) * masks_cropped).sum(-1).sum(-1) / torch.clamp(masks_cropped.sum(-1).sum(-1), min=1e-4)
                plane_parameters = plane_normals * offsets.view((-1, 1))
                pass
            pass
        detections = torch.cat([detections[:, :6], plane_parameters], dim=-1)
        pass
    return detections, masks

class ConvBlock(torch.nn.Module):
    """The block consists of a convolution layer, an optional batch normalization layer, and a ReLU layer"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, output_padding=0, mode='conv', use_bn=True):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn
        if mode == 'conv':
            self.conv = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv':
            self.conv = torch.nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        elif mode == 'upsample':
            self.conv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=stride, mode='nearest'), torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=padding, bias=not self.use_bn))
        elif mode == 'conv_3d':
            self.conv = torch.nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=not self.use_bn)
        elif mode == 'deconv_3d':
            self.conv = torch.nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=not self.use_bn)
        else:
            print('conv mode not supported', mode)
            exit(1)
            pass
        if '3d' not in mode:
            self.bn = torch.nn.BatchNorm2d(out_planes)
        else:
            self.bn = torch.nn.BatchNorm3d(out_planes)
            pass
        self.relu = torch.nn.ReLU(inplace=True)
        return
   
    def forward(self, inp):
        if self.use_bn:
            return self.relu(self.bn(self.conv(inp)))
        else:
            return self.relu(self.conv(inp))

class LinearBlock(torch.nn.Module):
    """The block consists of a linear layer and a ReLU layer"""    
    def __init__(self, in_planes, out_planes):
        super(LinearBlock, self).__init__()
        self.linear = torch.nn.Linear(in_planes, out_planes)
        self.relu = torch.nn.ReLU(inplace=True)
        return

    def forward(self, inp):
        return self.relu(self.linear(inp))       

def l2NormLossMask(pred, gt, mask, dim):
    """L2  loss with a mask"""
    return torch.sum(torch.norm(pred - gt, dim=dim) * mask) / torch.clamp(mask.sum(), min=1)

def l2LossMask(pred, gt, mask):
    """MSE with a mask"""    
    return torch.sum(torch.pow(pred - gt, 2) * mask) / torch.clamp(mask.sum(), min=1)

def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""        
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)

def invertDepth(depth, inverse=False):
    """Invert depth or not"""
    if inverse:
        valid_mask = (depth > 1e-4).float()
        depth_inv = 1.0 / torch.clamp(depth, min=1e-4)
        return depth_inv * valid_mask
    else:
        return depth

def grad_loss(grad_fake, grad_real):
    return torch.sum(torch.mean(torch.abs(grad_real - grad_fake)))

def imgrad(img):
    # input of size = [B, C, H, W]

    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)


####### 3D Reconstruction Modules

from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

def point_cloud(depth):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.

    """
    #fx = 756.8
    #fy = 756.0
    #centerX = 492.8
    #centerY = 270.4

    fx = 1170.187988
    fy = 1170.187988
    centerX = 647.750000
    centerY = 483.750000

    rows, cols = depth.shape
    c, r = torch.arange(cols).unsqueeze(0).cuda().float(), torch.arange(rows).unsqueeze(1).cuda().float()
    z = depth
    x = z * (c - centerX) / fx
    y = z * (r - centerY) / fy
    return torch.stack([x, y, z], dim=2)

def batch_pairwise_dist(x,y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    dtype = torch.cuda.LongTensor
    diag_ind_x = torch.arange(0, num_points_x).type(dtype)
    diag_ind_y = torch.arange(0, num_points_y).type(dtype)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2,1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P

def batch_NN_loss(x, y):

    # Set nan values to 0.
    x[torch.isnan(x)] = 0
    y[torch.isnan(y)] = 0

    bs, num_points, points_dim = x.size()
    dist1 = torch.sqrt(batch_pairwise_dist(x, y))
    values1, indices1 = dist1.min(dim=2)

    dist2 = torch.sqrt(batch_pairwise_dist(y, x))
    values2, indices2 = dist2.min(dim=2)
    a = torch.div(torch.sum(values1,1), num_points)
    b = torch.div(torch.sum(values2,1), num_points)
    sum = torch.div(torch.sum(a), bs) + torch.div(torch.sum(b), bs)

    return sum

def calculate_chamfer(gt_image, gt_depth, pred_depth, gt_masks, gt_boxes, gt_class_ids):
    rgb = gt_image[0].cpu().numpy().transpose(1, 2, 0)
    gt_depth = gt_depth[0].detach().cpu().numpy()
    pred_depth = pred_depth[0].detach().cpu().numpy()

    # print(rgb.shape)
    # expand the masks to full size, take first ex_i of 100 masks
    ex_i = torch.sum(gt_class_ids[0]!=0)
    expanded_mask = utils.expand_mask(gt_boxes[0].cpu().numpy(), gt_masks[0, :ex_i].cpu().numpy().transpose(1, 2, 0),
                                      rgb.shape)

    # print("Calculate point cloud gt...")
    # calculate the points from gt depth
    points_gt = point_cloud(gt_depth)

    # print("Calculate point cloud pred...")
    # calculate the points from pred depth
    points_pred = point_cloud(pred_depth)

    #points_gt = points_gt.reshape(-1, points_gt.shape[-1])
    #points_pred = points_pred.reshape(-1, points_pred.shape[-1])

    loss_sum = 0

    # print("Calculate chamfer loss...")
    # calculate the chamfer distance between the masked points
    for i in range(ex_i):
        # print(i)
        m = expanded_mask[:, :, i]
        inds = np.where(m == True)

        if len(inds[0]) == 0:
            continue

        # reduce the number of points: minimum of 2500 or 1/3rd of indices
        num_inds = min(2500, len(inds[0]) // 3)
        rand_inds = np.random.randint(len(inds[0]), size=num_inds)
        inds = (inds[0][rand_inds], inds[1][rand_inds])

        masked_points_gt = np.array(points_gt)[inds]
        masked_points_pred = np.array(points_pred)[inds]

        # add back batch dimension
        if len(masked_points_gt) != 0:
            masked_points_gt = torch.tensor(masked_points_gt).unsqueeze(0)
            masked_points_pred = torch.tensor(masked_points_pred).unsqueeze(0)

            # print("shapes: ", masked_points_gt.shape, masked_points_pred.shape)

            # dist1, dist2, _, _ = distChamfer(masked_points_gt, masked_points_pred)
            # loss_net = (torch.mean(dist1)) + (torch.mean(dist1))
            loss_net = batch_NN_loss(masked_points_gt, masked_points_pred)
            #print("loss: ", loss_net)
            loss_sum += loss_net

    #print("chamf loss ", loss_sum, loss_sum.dtype, " ex i:", ex_i)

    return torch.tensor(loss_sum).float() / ex_i.float().data.cpu().item()
    # return losses

def calculate_chamfer_scene(gt_depth, pred_depth):
    # calculate the points
    points_gt = point_cloud(gt_depth[0])
    points_pred = point_cloud(pred_depth[0])

    #points_gt = points_gt.reshape(-1, points_gt.shape[-1])
    #points_pred = points_pred.reshape(-1, points_pred.shape[-1])

    # Sample the points for quick calculation
    # num_inds = 300000
    # print(points_gt.shape)
    # inds = np.random.choice(points_gt.shape, num_inds)

    # print("inds after selection: ", inds[0].shape)

    # points_gt = points_gt[inds-1]
    # points_pred = points_pred[inds-1]

    # add back batch dimension
    #points_gt = points_gt.unsqueeze(0)
    #points_pred = points_pred.unsqueeze(0)
    # print("shapes: ", points_gt.shape, points_pred.shape)

    # Set nan values to 0
    #points_gt[torch.isnan(points_gt)] = 0
    #points_pred[torch.isnan(points_pred)] = 0
    # print('points_gt: ', points_gt.shape, points_gt[0, 50])

    # print("Calculate chamfer loss...")
    # calculate the chamfer distance
    # points_gt = Variable(points_gt, requires_grad=True)
    # points_pred = Variable(points_pred, requires_grad=True)

    # loss = batch_NN_loss(points_gt, points_pred)
    dist1, dist2 = chamfer_dist(points_gt.float(), points_pred.float())
    loss = (torch.mean(dist1)) + (torch.mean(dist2))

    # print("loss type: ", loss_sum.dtype)
    return loss
