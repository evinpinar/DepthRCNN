"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import random
import itertools
import numpy as np
import colorsys
from skimage.measure import find_contours
import cv2

from models.model import detection_layer, unmold_detections
from models.modules import *
from utils import *

import matplotlib.pyplot as plt
if "DISPLAY" not in os.environ:
    plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import utils

def tileImages(image_list, padding_x=5, padding_y=5, background_color=0):
    """Tile images"""
    height = image_list[0][0].shape[0]
    width = image_list[0][0].shape[1]
    result_image = np.full((height * len(image_list) + padding_y * (len(image_list) + 1), width * len(image_list[0]) + padding_x * (len(image_list[0]) + 1), 3), fill_value=background_color, dtype=np.uint8)
    for index_y, images in enumerate(image_list):
        for index_x, image in enumerate(images):
            offset_x = index_x * width + (index_x + 1) * padding_x
            offset_y = index_y * height + (index_y + 1) * padding_y
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1).tile((1, 1, 3))
                pass
            result_image[offset_y:offset_y + height, offset_x:offset_x + width] = image
            continue
        continue
    return result_image

############################################################
#  Batch visualization
############################################################
def visualizeBatchDeMoN(options, input_dict, results, indexOffset=0, prefix='', concise=False):
    cornerColorMap = {'gt': np.array([255, 0, 0]), 'pred': np.array([0, 0, 255]), 'inp': np.array([0, 255, 0])}
    topdownSize = 256
    
    for batchIndex in range(len(input_dict['image_1'])):
        pose = input_dict['pose'][batchIndex]

        for resultIndex, result in enumerate(results):
            if concise and resultIndex < len(results) - 1:
                continue
            depth_pred = invertDepth(result['depth'][batchIndex]).detach().cpu().numpy().squeeze()
            depth_gt = input_dict['depth'][batchIndex].squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_pred_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            
            
            if 'flow' in result:
                flow_pred = result['flow'][batchIndex, :2].detach().cpu().numpy().transpose((1, 2, 0))
                cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_flow_pred_' + str(len(results) - 1 - resultIndex) + '.png', cv2.resize(drawFlowImage(flow_pred), (256, 192)))
                pass
            if 'rotation' in result and resultIndex >= len(results) - 2:
                pass
            continue

        if not concise:
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_gt.png', drawDepthImage(input_dict['depth'][batchIndex]))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image_0.png', (input_dict['image_1'][batchIndex].transpose((1, 2, 0)) + 0.5) * 255)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image_1.png', (input_dict['image_2'][batchIndex].transpose((1, 2, 0)) + 0.5) * 255)
            flow_gt = input_dict['flow'][batchIndex, :2].transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_flow_gt.png', cv2.resize(drawFlowImage(flow_gt), (256, 192)))
            pass
        continue
    return

def visualizeBatchPair(options, config, inp_pair, detection_pair, indexOffset=0, prefix='', suffix='', write_ply=False, write_new_view=False):
    detection_images = []    
    for pair_index, (input_dict, detection_dict) in enumerate(zip(inp_pair, detection_pair)):
        image_dict = visualizeBatchDetection(options, config, input_dict, detection_dict, indexOffset=indexOffset, prefix=prefix, suffix='_' + str(pair_index), prediction_suffix=suffix, write_ply=write_ply, write_new_view=write_new_view)
        detection_images.append(image_dict['detection'])
        continue
    detection_image = tileImages([detection_images])
    return

def visualizeBatchRefinement(options, config, input_dict, results, indexOffset=0, prefix='', suffix='', concise=False):
    if not concise:
        image = (input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_0.png', image)
        image_2 = (input_dict['image_2'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_1.png', image_2)
        depth_gt = input_dict['depth'].detach().cpu().numpy().squeeze()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_gt.png', drawDepthImage(depth_gt))
        flow_gt = input_dict['flow'][0, :2].detach().cpu().numpy().transpose((1, 2, 0))
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_flow_gt.png', cv2.resize(drawFlowImage(flow_gt), (256, 192)))
        pass
    numbers = []
    for resultIndex, result in enumerate(results):
        if 'mask' in result and (options.losses == '' or '0' in options.losses):
            masks = result['mask'].detach().cpu().numpy()
            masks = np.concatenate([np.maximum(1 - masks.sum(0, keepdims=True), 0), masks], axis=0).transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation_' + str(len(results) - 1 - resultIndex) + '.png', drawSegmentationImage(masks, blackIndex=0) * (masks.max(-1, keepdims=True) > 0.5).astype(np.uint8))
            pass
        if concise:
            continue
        if 'depth' in result and (options.losses == '' or '3' in options.losses):
            depth_pred = invertDepth(result['depth']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        if 'plane_depth' in result and (options.losses == '' or '3' in options.losses):
            depth_pred = invertDepth(result['plane_depth']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass
            
            if options.scaleMode != 'variant':
                valid_mask = np.logical_and(depth_gt > 1e-4, depth_pred > 1e-4)
                depth_gt_values = depth_gt[valid_mask]
                depth_pred_values = depth_pred[valid_mask]
                scale = np.exp(np.mean(np.log(depth_gt_values) - np.log(depth_pred_values)))
                depth_pred *= scale
                pass
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_plane_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        if 'flow' in result and (options.losses == '' or '1' in options.losses):
            flow_pred = result['flow'][0, :2].detach().cpu().numpy().transpose((1, 2, 0))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_flow_pred_' + str(len(results) - 1 - resultIndex) + '.png', cv2.resize(drawFlowImage(flow_pred), (256, 192)))
            pass
        if 'rotation' in result and resultIndex >= len(results) - 2:
            pass
        if 'plane' in result and resultIndex > 0:
            numbers.append(np.linalg.norm(result['plane'].detach().cpu().numpy() - results[0]['plane'].detach().cpu().numpy()))
            pass
        if 'warped_image' in result and resultIndex >= len(results) - 2:
            warped_image = ((result['warped_image'].detach().cpu().numpy().transpose((0, 2, 3, 1))[0] + 0.5) * 255).astype(np.uint8)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image_warped_' + str(len(results) - 1 - resultIndex) + '.png', warped_image)
            pass

        if 'plane_depth_one_hot' in result:
            depth_pred = invertDepth(result['plane_depth_one_hot']).detach().cpu().numpy().squeeze()
            if depth_pred.shape[0] != depth_gt.shape[0]:
                depth_pred = cv2.resize(depth_pred, (depth_gt.shape[1], depth_gt.shape[0]))
                pass            
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_pred_plane_onehot_' + str(len(results) - 1 - resultIndex) + '.png', drawDepthImage(depth_pred))
            pass
        
        continue
    if 'parameter' in options.suffix:
        print('plane diff', numbers)
        pass
    return

def visualizeBatchDetection(options, config, input_dict, detection_dict, indexOffset=0, prefix='', suffix='', prediction_suffix='', write_ply=False, write_new_view=False):
    image_dict = {}
    images = input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
    images = unmold_image(images, config)
    image = images[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '.png', image[80:560])
    
    if 'warped_image' in input_dict:
        warped_images = input_dict['warped_image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
        warped_images = unmold_image(warped_images, config)
        warped_image = warped_images[0]
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '_warped.png', warped_image[80:560])
        pass

    if 'warped_depth' in input_dict:
        warped_depth = input_dict['warped_depth'].detach().cpu().numpy()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '_warped.png', drawDepthImage(warped_depth[80:560]))
        pass

    if 'warped_mask' in input_dict:
        warped_mask = input_dict['warped_mask'].detach().cpu().numpy()[0]
        pass

    if 'depth' in input_dict:
        depths = input_dict['depth'].detach().cpu().numpy()                
        depth_gt = depths[0]
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '.png', drawDepthImage(depth_gt[80:560]))
        pass

    windows = (0, 0, images.shape[1], images.shape[2])        
    windows = (0, 0, images.shape[1], images.shape[2])                
    class_colors = ColorPalette(config.NUM_CLASSES).getColorMap().tolist()        

    if 'mask' in input_dict:
        box_image = image.copy()
        boxes = input_dict['bbox'][0].detach().cpu().numpy()
        masks = input_dict['mask'][0].detach().cpu().numpy()
        if config.NUM_PARAMETER_CHANNELS > 0:
            depths = masks[:, :, :, 1]
            masks = masks[:, :, :, 0]
            pass

        segmentation_image = image * 0.0
        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
            segmentation_image[box[0]:box[2], box[1]:box[3]] = np.minimum(segmentation_image[box[0]:box[2], box[1]:box[3]] + np.expand_dims(mask, axis=-1) * np.random.randint(255, size=(3, ), dtype=np.int32), 255)
            continue
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation' + suffix + '.png', segmentation_image.astype(np.uint8)[80:560])
        if config.NUM_PARAMETER_CHANNELS > 0 and not config.OCCLUSION:
            depth_image = np.zeros((image.shape[0], image.shape[1]))
            for box, patch_depth in zip(boxes, depths):
                box = np.round(box).astype(np.int32)
                patch_depth = cv2.resize(patch_depth, (box[3] - box[1], box[2] - box[0]), cv2.INTER_NEAREST)
                depth_image[box[0]:box[2], box[1]:box[3]] = patch_depth
                continue
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth_patch' + suffix + '.png', drawDepthImage(depth_image[80:560]))
            pass
        pass

    if 'boundary' in detection_dict:
        boundary_pred = detection_dict['boundary'].detach().cpu().numpy()[0]
        boundary_gt = input_dict['boundary'].detach().cpu().numpy()[0]
        for name, boundary in [('gt', boundary_gt), ('pred', boundary_pred)]:
            boundary_image = image.copy()
            boundary_image[boundary[0] > 0.5] = np.array([255, 0, 0])
            boundary_image[boundary[1] > 0.5] = np.array([0, 0, 255])        
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_boundary' + suffix + '_' + name + '.png', boundary_image)
            continue
        pass
        
    if 'depth' in detection_dict:    
        depth_pred = detection_dict['depth'][0].detach().cpu().numpy()
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '.png', drawDepthImage(depth_pred[80:560]))                    
        if options.debug:
            valid_mask = (depth_gt > 1e-4) * (input_dict['segmentation'].detach().cpu().numpy()[0] >= 0) * (detection_dict['mask'].detach().cpu().numpy().squeeze() > 0.5)
            pass
        pass
    
    if 'depth_np' in detection_dict:
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '_np.png', drawDepthImage(detection_dict['depth_np'].squeeze().detach().cpu().numpy()[80:560]))
        pass

    if 'depth_ori' in detection_dict:
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + prediction_suffix + '_ori.png', drawDepthImage(detection_dict['depth_ori'].squeeze().detach().cpu().numpy()[80:560]))
        pass
    

    if 'detection' in detection_dict and len(detection_dict['detection']) > 0:
        detections = detection_dict['detection'].detach().cpu().numpy()
        detection_masks = detection_dict['masks'].detach().cpu().numpy().transpose((1, 2, 0))
        if 'flag' in detection_dict:
            detection_flags = detection_dict['flag']
        else:
            detection_flags = {}
            pass
        instance_image, normal_image, depth_image = draw_instances(config, image, depth_gt, detections[:, :4], detection_masks > 0.5, detections[:, 4].astype(np.int32), detections[:, 6:], detections[:, 5], draw_mask=True, transform_planes=False, detection_flags=detection_flags)
        image_dict['detection'] = instance_image
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_segmentation' + suffix + prediction_suffix + '.png', instance_image[80:560])
    else:
        image_dict['detection'] = np.zeros(image.shape, dtype=image.dtype)
        pass

    if write_new_view and False:
        detection_masks = detection_dict['masks']
        pose = np.eye(4)
        pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(np.array([-1, 0, 0]), np.pi / 18 * 0), axisAngleToRotationMatrix(np.array([0, 0, -1]), np.pi / 18))
        pose[:3, 3] = np.array([-0.4, 0, 0])        
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view' + suffix + prediction_suffix + '.png', detection_masks[:, 80:560].detach().cpu().numpy(), detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560], input_dict['camera'].detach().cpu().numpy(), pose)
        depth = depth_gt[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = ranges * np.expand_dims(depth, axis=-1)
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view_depth_gt' + suffix + prediction_suffix + '.png', np.expand_dims(depth > 1e-4, 0), np.expand_dims(XYZ_gt, 0), input_dict['camera'].detach().cpu().numpy(), pose)
        depth = detection_dict['depth_np'].squeeze()[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = ranges * np.expand_dims(depth, axis=-1)
        drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_new_view_depth_pred' + suffix + prediction_suffix + '.png', np.expand_dims(depth > 1e-4, 0), np.expand_dims(XYZ_gt, 0), input_dict['camera'].detach().cpu().numpy(), pose)        
        pass

    if write_new_view:
        detection_masks = detection_dict['masks'][:, 80:560].detach().cpu().numpy()
        XYZ_pred = detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560]
        depth = depth_gt[80:560]
        ranges = config.getRanges(input_dict['camera']).detach().cpu().numpy()
        XYZ_gt = np.expand_dims(ranges * np.expand_dims(depth, axis=-1), 0)
        valid_mask = np.expand_dims(depth > 1e-4, 0).astype(np.float32)
        camera = input_dict['camera'].detach().cpu().numpy()

        valid_mask = np.expand_dims(cv2.resize(valid_mask[0], (256, 192)), 0)
        XYZ_gt = np.expand_dims(cv2.resize(XYZ_gt[0], (256, 192)), 0)
        detection_masks = np.stack([cv2.resize(detection_masks[c], (256, 192)) for c in range(len(detection_masks))], axis=0)
        XYZ_pred = np.stack([cv2.resize(XYZ_pred[c], (256, 192)) for c in range(len(XYZ_pred))], axis=0)
        locations = [np.array([-0.4, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.4, 0, 0])]
        angle_pairs = [(np.array([-1, 0, 0, np.pi / 18 * 0]), np.array([0, 0, -1, np.pi / 18])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([-1, 0, 0, np.pi / 18 * 0]), np.array([0, 0, 1, np.pi / 18]))]
        num_frames = [25, 10, 25]
        for c in range(len(locations) - 1):
            if c == 2:
                continue
            for frame in range(num_frames[c]):
                ratio = float(frame + 1) / num_frames[c]
                location = locations[c] + (locations[c + 1] - locations[c]) * ratio
                angle_pair = [angle_pairs[c][dim] + (angle_pairs[c + 1][dim] - angle_pairs[c][dim]) * ratio for dim in range(2)]
                pose = np.eye(4)
                pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(angle_pair[0][:3], angle_pair[0][3]), axisAngleToRotationMatrix(angle_pair[1][:3], angle_pair[1][3]))
                pose[:3, 3] = location

                index_offset = sum(num_frames[:c]) + frame
                drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_video/' + str(index_offset) + '.png', detection_masks, XYZ_pred, camera, pose)
                
                drawNewViewDepth(options.test_dir + '/' + str(indexOffset) + '_video_gt/' + str(index_offset) + '.png', valid_mask, XYZ_gt, camera, pose)
                continue
            continue
        exit(1)
        pass
    
    if write_ply:
        detection_masks = detection_dict['masks']
        if 'plane_XYZ' not in detection_dict:
            plane_XYZ = planeXYZModule(config.getRanges(input_dict['camera']), detection_dict['detection'][:, 6:9], width=config.IMAGE_MAX_DIM, height=config.IMAGE_MIN_DIM)
            plane_XYZ = plane_XYZ.transpose(1, 2).transpose(0, 1).transpose(2, 3).transpose(1, 2)
            zeros = torch.zeros(int(plane_XYZ.shape[0]), 3, (config.IMAGE_MAX_DIM - config.IMAGE_MIN_DIM) // 2, config.IMAGE_MAX_DIM).cuda()
            plane_XYZ = torch.cat([zeros, plane_XYZ, zeros], dim=2)
            detection_dict['plane_XYZ'] = plane_XYZ
            pass

        print(options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply')
        writePLYFileMask(options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply', image[80:560], detection_masks[:, 80:560].detach().cpu().numpy(), detection_dict['plane_XYZ'].detach().cpu().numpy().transpose((0, 2, 3, 1))[:, 80:560], write_occlusion='occlusion' in options.suffix)

        pose = np.eye(4)
        pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(np.array([-1, 0, 0]), np.pi / 18), axisAngleToRotationMatrix(np.array([0, -1, 0]), np.pi / 18))
        pose[:3, 3] = np.array([-0.4, 0.3, 0])

        current_dir = os.path.dirname(os.path.realpath(__file__))
        pose_filename = current_dir + '/test/pose_new_view.txt'
        print(pose_filename)
        with open(pose_filename, 'w') as f:
            for row in pose:
                for col in row:
                    f.write(str(col) + '\t')
                    continue
                f.write('\n')
                continue
            f.close()
            pass
        model_filename = current_dir + '/' + options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.ply'
        output_filename = current_dir + '/' + options.test_dir + '/' + str(indexOffset) + '_model' + suffix + prediction_suffix + '.png'
        try:
            os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
        except:
            pass
        pass
    return image_dict


def visualizeBatchDepth(options, config, input_dict, detection_dict, indexOffset=0, prefix='', suffix='', write_ply=False):
    image_dict = {}
    images = input_dict['image'].detach().cpu().numpy().transpose((0, 2, 3, 1))
    images = unmold_image(images, config)
    for batchIndex, image in enumerate(images):
        cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_image' + suffix + '.png', image)
        continue

    depths = input_dict['depth'].detach().cpu().numpy()
    for batchIndex, depth in enumerate(depths):
        cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth' + suffix + '.png', drawDepthImage(depth))
        continue

    if 'depth_np' in detection_dict:
        for batchIndex, depth in enumerate(detection_dict['depth_np'].detach().cpu().numpy()):
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_depth_pred_np' + suffix + '.png', drawDepthImage(depth))
            continue
        pass
    
    return

def visualizeBatchSingle(options, config, images, image_metas, rpn_rois, depths, dicts, input_dict={}, inference={}, indexOffset=0, prefix='', suffix='', compare_planenet=False):

    image = images[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_image' + suffix + '.png', image)

    depth = depths[0]
    cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '.png', drawDepthImage(depth))

    windows = (0, 0, images.shape[1], images.shape[2])
    class_colors = ColorPalette(config.NUM_CLASSES).getColorMap(returnTuples=True)
    instance_colors = ColorPalette(1000).getColorMap(returnTuples=True)

    if 'mask' in input_dict:
        box_image = image.copy()
        boxes = input_dict['bbox'][0].detach().cpu().numpy()
        masks = input_dict['mask'][0].detach().cpu().numpy()

        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=(0, 0, 255), thickness=2)
            continue
        
        segmentation_image = image * 0.0
        for box, mask in zip(boxes, masks):
            box = np.round(box).astype(np.int32)
            mask = cv2.resize(mask, (box[3] - box[1], box[2] - box[0]))
            segmentation_image[box[0]:box[2], box[1]:box[3]] = np.minimum(segmentation_image[box[0]:box[2], box[1]:box[3]] + np.expand_dims(mask, axis=-1) * np.random.randint(255, size=(3, ), dtype=np.int32), 255)
            continue
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '.png', segmentation_image.astype(np.uint8))
        pass
    
    for name, result_dict in dicts:

        if len(rpn_rois) > 0:
            detections, keep_indices, ori_rois = detection_layer(config, rpn_rois.unsqueeze(0), result_dict['mrcnn_class'], result_dict['mrcnn_bbox'], result_dict['mrcnn_parameter'], image_metas, return_indices=True)
            box_image = image.copy()
            for instance_index, box in enumerate(detections.detach().cpu().numpy().astype(np.int32)):
                cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=class_colors[int(box[4])], thickness=3)
                continue
        else:
            continue
        
        if len(detections) > 0:
            detections[:, :4] = ori_rois

            detections = detections.detach().cpu().numpy()
            mrcnn_mask = result_dict['mrcnn_mask'][keep_indices].detach().cpu().numpy()

            if name == 'gt':
                class_mrcnn_mask = np.zeros(list(mrcnn_mask.shape) + [config.NUM_CLASSES], dtype=np.float32)
                for index, (class_id, mask) in enumerate(zip(detections[:, 4].astype(np.int32), mrcnn_mask)):
                    if config.GLOBAL_MASK:
                        class_mrcnn_mask[index, :, :, 0] = mask
                    else:
                        class_mrcnn_mask[index, :, :, class_id] = mask
                        pass
                    continue
                mrcnn_mask = class_mrcnn_mask
            else:
                mrcnn_mask = mrcnn_mask.transpose((0, 2, 3, 1))
                pass

            box_image = image.copy()
            for instance_index, box in enumerate(detections.astype(np.int32)):
                cv2.rectangle(box_image, (box[1], box[0]), (box[3], box[2]), color=tuple(class_colors[int(box[4])]), thickness=3)
                continue
            
            final_rois, final_class_ids, final_scores, final_masks, final_parameters = unmold_detections(config, detections, mrcnn_mask, image.shape, windows, debug=False)

            result = {
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "parameters": final_parameters,
            }


            instance_image, normal_image, depth_image = draw_instances(config, image, depth, result['rois'], result['masks'], result['class_ids'], result['parameters'], result['scores'])
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '_' + name + '.png', instance_image)
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_depth' + suffix + '_' + name + '.png', depth_image)
        else:
            print('no detections')
            pass
        continue


    if len(inference) > 0:
        instance_image, normal_image, depth_image = draw_instances(config, image, depth, inference['rois'], inference['masks'], inference['class_ids'], inference['parameters'], inference['scores'], draw_mask=True)
        cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_detection' + suffix + '.png', instance_image)

        if compare_planenet:
            print(image.shape, image.min(), image.max())
            pred_dict = detector.detect(image[80:560])
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_planenet_segmentation.png', drawSegmentationImage(pred_dict['segmentation'], blackIndex=10))
            cv2.imwrite(options.test_dir + '/' + str(indexOffset) + '_planenet_depth.png', drawDepthImage(pred_dict['depth']))
            pass
        pass
    return

def visualizeBatchBoundary(options, config, images, boundary_pred, boundary_gt, indexOffset=0):
    images = (images.detach().cpu().numpy().transpose((0, 2, 3, 1)) + config.MEAN_PIXEL).astype(np.uint8)
    boundary_pred = boundary_pred.detach().cpu().numpy()
    boundary_gt = boundary_gt.detach().cpu().numpy()
    for batchIndex in range(len(images)):
        for name, boundary in [('gt', boundary_gt[batchIndex]), ('pred', boundary_pred[batchIndex])]:
            image = images[batchIndex].copy()
            image[boundary[0] > 0.5] = np.array([255, 0, 0])
            image[boundary[1] > 0.5] = np.array([0, 0, 255])        
            cv2.imwrite(options.test_dir + '/' + str(indexOffset + batchIndex) + '_boundary_' + name + '.png', image)
            continue
        continue
    return

############################################################
#  Visualization
############################################################


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  np.minimum(image[:, :, c] *
                                             (1 - alpha) + alpha * color[c], 255),
                                  image[:, :, c])
    return image




def draw_instances(config, image, depth, boxes, masks, class_ids, parameters,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, draw_mask=False, transform_planes=False, statistics=[], detection_flags={}):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    ## Number of instances
    N = len(boxes)
    if not N:
        pass
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    ## Generate random colors
    instance_colors = ColorPalette(N).getColorMap(returnTuples=True)
    if len(detection_flags) and False:
        for index in range(N):
            if detection_flags[index] < 0.5:
                instance_colors[index] = (128, 128, 128)
                pass
            continue
        pass

    class_colors = ColorPalette(11).getColorMap(returnTuples=True)
    class_colors[0] = (128, 128, 128)
    
    ## Show area outside image boundaries.
    height, width = image.shape[:2]
    masked_image = image.astype(np.uint8).copy()
    normal_image = np.zeros(image.shape)
    depth_image = depth.copy()

    for i in range(N):

        ## Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        
        ## Label
        class_id = class_ids[i]

        score = scores[i] if scores is not None else None
        x = random.randint(x1, (x1 + x2) // 2)

        ## Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image.astype(np.float32), mask, instance_colors[i]).astype(np.uint8)
        
        ## Mask Polygon
        ## Pad to ensure proper polygons for masks that touch image edges.
        if draw_mask:
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                ## Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                cv2.polylines(masked_image, np.expand_dims(verts.astype(np.int32), 0), True, color=class_colors[class_id])
                continue

        continue
    
    normal_image = drawNormalImage(normal_image)    
    depth_image = drawDepthImage(depth_image)
    return masked_image.astype(np.uint8), normal_image.astype(np.uint8), depth_image




## Write the reconstruction result to PLY file
def writePLYFileMask(filename, image, masks, plane_XYZ, write_occlusion=False):

    width = image.shape[1]
    height = image.shape[0]
    
    betweenRegionThreshold = 0.1
    nonPlanarRegionThreshold = 0.02
    dotThreshold = np.cos(np.deg2rad(30))

    faces = []
    points = []

    masks = np.round(masks)
    plane_depths = plane_XYZ[:, :, :, 1] * masks + 10 * (1 - masks)
    segmentation = plane_depths.argmin(0)
    for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
        indices = np.nonzero(mask > 0.5)
        for y, x in zip(indices[0], indices[1]):
            if y == height - 1 or x == width - 1:
                continue
            validNeighborPixels = []
            for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    pass
                continue
            if len(validNeighborPixels) == 3:
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (x + 1, y + 1), (x + 1, y)]]
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (x, y + 1), (x + 1, y + 1)]]
            elif len(validNeighborPixels) == 2:
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for pixel in [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]), (validNeighborPixels[1][0], validNeighborPixels[1][1])]]
                pass
            continue
        continue

    imageFilename = "textureless"
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0"""
        header += imageFilename
        header += """
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
end_header
"""
        f.write(header)
        for point in points:
            X = point[0][0]
            Y = point[0][1]
            Z = point[0][2]
            if not write_occlusion or point[2]:
                color = image[point[1][1], point[1][0]]
            else:
                color = (128, 128, 128)
                pass
            f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
            continue

        for face in faces:
            valid = True
            f.write('3 ')                
            for c in face:
                f.write(str(c) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


def drawNewViewDepth(depth_filename, masks, XYZs, camera, pose):
    faces = []
    width, height = masks.shape[2], masks.shape[1]
    for mask, XYZ in zip(masks, XYZs):
        indices = np.nonzero(mask > 0.5)
        for y, x in zip(indices[0], indices[1]):
            if y == height - 1 or x == width - 1:
                continue
            validNeighborPixels = []
            for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    pass
                continue
            if len(validNeighborPixels) == 3:
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (x + 1, y + 1), (x + 1, y)]])
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (x, y + 1), (x + 1, y + 1)]])
            elif len(validNeighborPixels) == 2:
                faces.append([XYZ[pixel[1], pixel[0]] for pixel in [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]), (validNeighborPixels[1][0], validNeighborPixels[1][1])]])
                pass
            continue
        continue
    faces = np.array(faces)
    XYZ = faces.reshape((-1, 3))
    XYZ = np.matmul(np.concatenate([XYZ, np.ones((len(XYZ), 1))], axis=-1), pose.transpose())
    XYZ = XYZ[:, :3] / XYZ[:, 3:]
    points = XYZ[:, :3]
    depth = XYZ[:, 1:2]
    depth = np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(255 - depth, colormap=cv2.COLORMAP_JET).reshape((-1, 3))

    imageFilename = "textureless"
    filename = 'test/model.ply'
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0"""
        header += imageFilename
        header += """
element vertex """
        header += str(len(points))
        header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
        header += str(len(points) // 3)
        header += """
property list uchar int vertex_indices
end_header
"""
        f.write(header)
        for point, color in zip(points, colors):
            X = point[0]
            Y = point[1]
            Z = point[2]
            f.write(str(X) + ' ' +    str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n')
            continue

        faces = np.arange(len(points)).reshape((-1, 3))
        for face in faces:
            valid = True
            f.write('3 ')                
            for c in face:
                f.write(str(c) + ' ')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    pose = np.eye(4)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pose_filename = current_dir + '/test/pose_new_view.txt'
    with open(pose_filename, 'w') as f:
        for row in pose:
            for col in row:
                f.write(str(col) + '\t')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    model_filename = current_dir + '/test/model.ply'
    output_filename = current_dir + '/' + depth_filename
    try:
        os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
    except:
        print('depth rendering failed')
        pass
    return

def rotateModel(model_filename, output_folder):
    locations = [np.array([-0.4, 0.3, 0]), np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.4, 0.3, 0])]
    angle_pairs = [(np.array([-1, 0, 0, np.pi / 18]), np.array([0, -1, 0, np.pi / 18])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])), (np.array([-1, 0, 0, np.pi / 18]), np.array([0, 1, 0, np.pi / 18]))]
    num_frames = [50, 20, 50]
    for c in range(len(locations) - 1):
        for frame in range(num_frames[c]):
            ratio = float(frame + 1) / num_frames[c]
            location = locations[c] + (locations[c + 1] - locations[c]) * ratio
            angle_pair = [angle_pairs[c][dim] + (angle_pairs[c + 1][dim] - angle_pairs[c][dim]) * ratio for dim in range(2)]
            pose = np.eye(4)
            pose[:3, :3] = np.matmul(axisAngleToRotationMatrix(angle_pair[0][:3], angle_pair[0][3]), axisAngleToRotationMatrix(angle_pair[1][:3], angle_pair[1][3]))
            pose[:3, 3] = location
            current_dir = os.path.dirname(os.path.realpath(__file__))
            pose_filename = output_folder + '/%04d'%(sum(num_frames[:c]) + frame) + '.txt'
            with open(pose_filename, 'w') as f:
                for row in pose:
                    for col in row:
                        f.write(str(col) + '\t')
                        continue
                    f.write('\n')
                    continue
                f.close()
                pass
            continue
        continue
    try:
        os.system('../../../Recorder/Recorder --model_filename=' + model_filename + ' --output_folder=' + output_folder + ' --pose_folder=' + output_folder + ' --num_frames=' + str(sum(num_frames)))
    except:
        print('Recording failed')
        pass
    pass

def visualizeGraph(var, params):
    """Visualize the network"""
    from torchviz import make_dot
    return make_dot(var, params)

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    plt.show()


turbo_colormap_data = [[0.18995,0.07176,0.23217],[0.19483,0.08339,0.26149],[0.19956,0.09498,0.29024],[0.20415,0.10652,0.31844],[0.20860,0.11802,0.34607],[0.21291,0.12947,0.37314],[0.21708,0.14087,0.39964],[0.22111,0.15223,0.42558],[0.22500,0.16354,0.45096],[0.22875,0.17481,0.47578],[0.23236,0.18603,0.50004],[0.23582,0.19720,0.52373],[0.23915,0.20833,0.54686],[0.24234,0.21941,0.56942],[0.24539,0.23044,0.59142],[0.24830,0.24143,0.61286],[0.25107,0.25237,0.63374],[0.25369,0.26327,0.65406],[0.25618,0.27412,0.67381],[0.25853,0.28492,0.69300],[0.26074,0.29568,0.71162],[0.26280,0.30639,0.72968],[0.26473,0.31706,0.74718],[0.26652,0.32768,0.76412],[0.26816,0.33825,0.78050],[0.26967,0.34878,0.79631],[0.27103,0.35926,0.81156],[0.27226,0.36970,0.82624],[0.27334,0.38008,0.84037],[0.27429,0.39043,0.85393],[0.27509,0.40072,0.86692],[0.27576,0.41097,0.87936],[0.27628,0.42118,0.89123],[0.27667,0.43134,0.90254],[0.27691,0.44145,0.91328],[0.27701,0.45152,0.92347],[0.27698,0.46153,0.93309],[0.27680,0.47151,0.94214],[0.27648,0.48144,0.95064],[0.27603,0.49132,0.95857],[0.27543,0.50115,0.96594],[0.27469,0.51094,0.97275],[0.27381,0.52069,0.97899],[0.27273,0.53040,0.98461],[0.27106,0.54015,0.98930],[0.26878,0.54995,0.99303],[0.26592,0.55979,0.99583],[0.26252,0.56967,0.99773],[0.25862,0.57958,0.99876],[0.25425,0.58950,0.99896],[0.24946,0.59943,0.99835],[0.24427,0.60937,0.99697],[0.23874,0.61931,0.99485],[0.23288,0.62923,0.99202],[0.22676,0.63913,0.98851],[0.22039,0.64901,0.98436],[0.21382,0.65886,0.97959],[0.20708,0.66866,0.97423],[0.20021,0.67842,0.96833],[0.19326,0.68812,0.96190],[0.18625,0.69775,0.95498],[0.17923,0.70732,0.94761],[0.17223,0.71680,0.93981],[0.16529,0.72620,0.93161],[0.15844,0.73551,0.92305],[0.15173,0.74472,0.91416],[0.14519,0.75381,0.90496],[0.13886,0.76279,0.89550],[0.13278,0.77165,0.88580],[0.12698,0.78037,0.87590],[0.12151,0.78896,0.86581],[0.11639,0.79740,0.85559],[0.11167,0.80569,0.84525],[0.10738,0.81381,0.83484],[0.10357,0.82177,0.82437],[0.10026,0.82955,0.81389],[0.09750,0.83714,0.80342],[0.09532,0.84455,0.79299],[0.09377,0.85175,0.78264],[0.09287,0.85875,0.77240],[0.09267,0.86554,0.76230],[0.09320,0.87211,0.75237],[0.09451,0.87844,0.74265],[0.09662,0.88454,0.73316],[0.09958,0.89040,0.72393],[0.10342,0.89600,0.71500],[0.10815,0.90142,0.70599],[0.11374,0.90673,0.69651],[0.12014,0.91193,0.68660],[0.12733,0.91701,0.67627],[0.13526,0.92197,0.66556],[0.14391,0.92680,0.65448],[0.15323,0.93151,0.64308],[0.16319,0.93609,0.63137],[0.17377,0.94053,0.61938],[0.18491,0.94484,0.60713],[0.19659,0.94901,0.59466],[0.20877,0.95304,0.58199],[0.22142,0.95692,0.56914],[0.23449,0.96065,0.55614],[0.24797,0.96423,0.54303],[0.26180,0.96765,0.52981],[0.27597,0.97092,0.51653],[0.29042,0.97403,0.50321],[0.30513,0.97697,0.48987],[0.32006,0.97974,0.47654],[0.33517,0.98234,0.46325],[0.35043,0.98477,0.45002],[0.36581,0.98702,0.43688],[0.38127,0.98909,0.42386],[0.39678,0.99098,0.41098],[0.41229,0.99268,0.39826],[0.42778,0.99419,0.38575],[0.44321,0.99551,0.37345],[0.45854,0.99663,0.36140],[0.47375,0.99755,0.34963],[0.48879,0.99828,0.33816],[0.50362,0.99879,0.32701],[0.51822,0.99910,0.31622],[0.53255,0.99919,0.30581],[0.54658,0.99907,0.29581],[0.56026,0.99873,0.28623],[0.57357,0.99817,0.27712],[0.58646,0.99739,0.26849],[0.59891,0.99638,0.26038],[0.61088,0.99514,0.25280],[0.62233,0.99366,0.24579],[0.63323,0.99195,0.23937],[0.64362,0.98999,0.23356],[0.65394,0.98775,0.22835],[0.66428,0.98524,0.22370],[0.67462,0.98246,0.21960],[0.68494,0.97941,0.21602],[0.69525,0.97610,0.21294],[0.70553,0.97255,0.21032],[0.71577,0.96875,0.20815],[0.72596,0.96470,0.20640],[0.73610,0.96043,0.20504],[0.74617,0.95593,0.20406],[0.75617,0.95121,0.20343],[0.76608,0.94627,0.20311],[0.77591,0.94113,0.20310],[0.78563,0.93579,0.20336],[0.79524,0.93025,0.20386],[0.80473,0.92452,0.20459],[0.81410,0.91861,0.20552],[0.82333,0.91253,0.20663],[0.83241,0.90627,0.20788],[0.84133,0.89986,0.20926],[0.85010,0.89328,0.21074],[0.85868,0.88655,0.21230],[0.86709,0.87968,0.21391],[0.87530,0.87267,0.21555],[0.88331,0.86553,0.21719],[0.89112,0.85826,0.21880],[0.89870,0.85087,0.22038],[0.90605,0.84337,0.22188],[0.91317,0.83576,0.22328],[0.92004,0.82806,0.22456],[0.92666,0.82025,0.22570],[0.93301,0.81236,0.22667],[0.93909,0.80439,0.22744],[0.94489,0.79634,0.22800],[0.95039,0.78823,0.22831],[0.95560,0.78005,0.22836],[0.96049,0.77181,0.22811],[0.96507,0.76352,0.22754],[0.96931,0.75519,0.22663],[0.97323,0.74682,0.22536],[0.97679,0.73842,0.22369],[0.98000,0.73000,0.22161],[0.98289,0.72140,0.21918],[0.98549,0.71250,0.21650],[0.98781,0.70330,0.21358],[0.98986,0.69382,0.21043],[0.99163,0.68408,0.20706],[0.99314,0.67408,0.20348],[0.99438,0.66386,0.19971],[0.99535,0.65341,0.19577],[0.99607,0.64277,0.19165],[0.99654,0.63193,0.18738],[0.99675,0.62093,0.18297],[0.99672,0.60977,0.17842],[0.99644,0.59846,0.17376],[0.99593,0.58703,0.16899],[0.99517,0.57549,0.16412],[0.99419,0.56386,0.15918],[0.99297,0.55214,0.15417],[0.99153,0.54036,0.14910],[0.98987,0.52854,0.14398],[0.98799,0.51667,0.13883],[0.98590,0.50479,0.13367],[0.98360,0.49291,0.12849],[0.98108,0.48104,0.12332],[0.97837,0.46920,0.11817],[0.97545,0.45740,0.11305],[0.97234,0.44565,0.10797],[0.96904,0.43399,0.10294],[0.96555,0.42241,0.09798],[0.96187,0.41093,0.09310],[0.95801,0.39958,0.08831],[0.95398,0.38836,0.08362],[0.94977,0.37729,0.07905],[0.94538,0.36638,0.07461],[0.94084,0.35566,0.07031],[0.93612,0.34513,0.06616],[0.93125,0.33482,0.06218],[0.92623,0.32473,0.05837],[0.92105,0.31489,0.05475],[0.91572,0.30530,0.05134],[0.91024,0.29599,0.04814],[0.90463,0.28696,0.04516],[0.89888,0.27824,0.04243],[0.89298,0.26981,0.03993],[0.88691,0.26152,0.03753],[0.88066,0.25334,0.03521],[0.87422,0.24526,0.03297],[0.86760,0.23730,0.03082],[0.86079,0.22945,0.02875],[0.85380,0.22170,0.02677],[0.84662,0.21407,0.02487],[0.83926,0.20654,0.02305],[0.83172,0.19912,0.02131],[0.82399,0.19182,0.01966],[0.81608,0.18462,0.01809],[0.80799,0.17753,0.01660],[0.79971,0.17055,0.01520],[0.79125,0.16368,0.01387],[0.78260,0.15693,0.01264],[0.77377,0.15028,0.01148],[0.76476,0.14374,0.01041],[0.75556,0.13731,0.00942],[0.74617,0.13098,0.00851],[0.73661,0.12477,0.00769],[0.72686,0.11867,0.00695],[0.71692,0.11268,0.00629],[0.70680,0.10680,0.00571],[0.69650,0.10102,0.00522],[0.68602,0.09536,0.00481],[0.67535,0.08980,0.00449],[0.66449,0.08436,0.00424],[0.65345,0.07902,0.00408],[0.64223,0.07380,0.00401],[0.63082,0.06868,0.00401],[0.61923,0.06367,0.00410],[0.60746,0.05878,0.00427],[0.59550,0.05399,0.00453],[0.58336,0.04931,0.00486],[0.57103,0.04474,0.00529],[0.55852,0.04028,0.00579],[0.54583,0.03593,0.00638],[0.53295,0.03169,0.00705],[0.51989,0.02756,0.00780],[0.50664,0.02354,0.00863],[0.49321,0.01963,0.00955],[0.47960,0.01583,0.01055]]
turbo_colormap_data_np = np.array(turbo_colormap_data)

# The look-up table contains 256 entries. Each entry is a floating point sRGB triplet.
# To use it with matplotlib, pass cmap=ListedColormap(turbo_colormap_data) as an arg to imshow() (don't forget "from matplotlib.colors import ListedColormap").
# If you have a typical 8-bit greyscale image, you can use the 8-bit value to index into this LUT directly.
# The floating point color values can be converted to 8-bit sRGB via multiplying by 255 and casting/flooring to an integer. Saturation should not be required for IEEE-754 compliant arithmetic.
# If you have a floating point value in the range [0,1], you can use interpolate() to linearly interpolate between the entries.
# If you have 16-bit or 32-bit integer values, convert them to floating point values on the [0,1] range and then use interpolate(). Doing the interpolation in floating point will reduce banding.
# If some of your values may lie outside the [0,1] range, use interpolate_or_clip() to highlight them.

def normalize_depth(depth_map):
    p_min = depth_map.min()
    p_max = depth_map.max()
    depth_normal = (depth_map - p_min) / (p_max - p_min)
    return depth_normal

def heatmap_to_pseudo_color(heatmap):
    x = heatmap
    x = x.clip(0, 1)
    a = (x * 255).astype(int)
    b = (a + 1).clip(max=255)
    f = x * 255.0 - a
    pseudo_color = (
        turbo_colormap_data_np[a]
        + (turbo_colormap_data_np[b] - turbo_colormap_data_np[a]) * f[..., None]
    )
    pseudo_color[heatmap < 0.0] = 0.0
    pseudo_color[heatmap > 1.0] = 1.0
    return pseudo_color

def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
            [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
        else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def plot_loss(loss, val_loss, save=True, log_dir=None):
    loss = np.array(loss)
    val_loss = np.array(val_loss)

    plt.figure("loss")
    plt.gcf().clear()
    plt.plot(loss[:, 0], label='train')
    plt.plot(val_loss[:, 0], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("rpn_class_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 1], label='train')
    plt.plot(val_loss[:, 1], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "rpn_class_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("rpn_bbox_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 2], label='train')
    plt.plot(val_loss[:, 2], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "rpn_bbox_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_class_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 3], label='train')
    plt.plot(val_loss[:, 3], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_class_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_bbox_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 4], label='train')
    plt.plot(val_loss[:, 4], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_bbox_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)

    plt.figure("mrcnn_mask_loss")
    plt.gcf().clear()
    plt.plot(loss[:, 5], label='train')
    plt.plot(val_loss[:, 5], label='valid')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    if save:
        save_path = os.path.join(log_dir, "mrcnn_mask_loss.png")
        plt.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.1)


if __name__ == '__main__':
    pose = np.eye(4)
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    pose_filename = current_dir + '/test/pose.txt'
    with open(pose_filename, 'w') as f:
        for row in pose:
            for col in row:
                f.write(str(col) + '\t')
                continue
            f.write('\n')
            continue
        f.close()
        pass
    test_dir = 'test/occlusion_debug'
    indexOffset = 33
    model_filename = current_dir + '/test/model.ply'
    output_filename = current_dir + '/' + test_dir + '/' + str(indexOffset) + '_model_0_occlusion.png'
    print('screenshot', output_filename)
    os.system('../../../Screenshoter/Screenshoter --model_filename=' + model_filename + ' --output_filename=' + output_filename + ' --pose_filename=' + pose_filename)
    exit(1)    
