"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
from torch import optim
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import numpy as np
import cv2
import sys

from models.model import *
from models.refinement_net import *
from models.modules import *
from datasets.plane_stereo_dataset import *

from utils import *
from visualize_utils import *
from evaluate_utils import *
from options import parse_args
from config import PlaneConfig

    
def train(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    config = PlaneConfig(options)
    
    dataset = PlaneDataset(options, config, split='train', random=True)
    dataset_test = PlaneDataset(options, config, split='test', random=False)

    print('the number of images', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=16)

    model = MaskRCNN(config)
    model.cuda()
    model.train()

    if options.restore == 1:
        ## Resume training
        print('restore')
        model.load_state_dict(torch.load(options.checkpoint_dir + '/checkpoint.pth'))
    elif options.restore == 2:
        ## Train upon Mask R-CNN weights
        model_path = options.MaskRCNNPath
        print("Loading pretrained weights ", model_path)
        model.load_weights(model_path)
        pass
    
    if options.trainingMode != '':
        ## Specify which layers to train, default is "all"
        layer_regex = {
            ## all layers but the backbone
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            ## From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            ## All layers
            "all": ".*",
            "classifier": "(classifier.*)|(mask.*)|(depth.*)",
        }
        assert(options.trainingMode in layer_regex.keys())
        layers = layer_regex[options.trainingMode]
        model.set_trainable(layers)
        pass

    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]

    model_names = [name for name, param in model.named_parameters()]

    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=options.LR, momentum=0.9)
    
    if options.restore == 1 and os.path.exists(options.checkpoint_dir + '/optim.pth'):
        optimizer.load_state_dict(torch.load(options.checkpoint_dir + '/optim.pth'))        
        pass

    
    for epoch in range(options.numEpochs):
        epoch_losses = []
        data_iterator = tqdm(dataloader, total=len(dataset) + 1)

        optimizer.zero_grad()

        for sampleIndex, sample in enumerate(data_iterator):
            losses = []            

            input_pair = []
            detection_pair = []
            dicts_pair = []

            camera = sample[30][0].cuda()                
            for indexOffset in [0, 13]:
                images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, gt_depth, extrinsics, gt_plane, gt_segmentation, plane_indices = sample[indexOffset + 0].cuda(), sample[indexOffset + 1].numpy(), sample[indexOffset + 2].cuda(), sample[indexOffset + 3].cuda(), sample[indexOffset + 4].cuda(), sample[indexOffset + 5].cuda(), sample[indexOffset + 6].cuda(), sample[indexOffset + 8].cuda(), sample[indexOffset + 9].cuda(), sample[indexOffset + 10].cuda(), sample[indexOffset + 11].cuda(), sample[indexOffset + 12].cuda()

                if indexOffset == 13:
                    input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'plane': gt_plane})
                    continue
                rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask, detections, detection_masks, detection_gt_masks, rpn_rois, roi_features, roi_indices, feature_map, depth_np_pred = model.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training_detection', use_nms=2, use_refinement='refinement' in options.suffix, return_feature_map=True)

                rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss, _ = compute_losses(config, rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)

                losses += [rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss]


                depth_np_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                losses.append(depth_np_loss)
                normal_np_pred = None

                if len(detections) > 0:
                    detections, detection_masks = unmoldDetections(config, detections, detection_masks, depth_np_pred, normal_np_pred, debug=False)
                    XYZ_pred, detection_mask, plane_XYZ = calcXYZModule(config, camera, detections, detection_masks, depth_np_pred, return_individual=True)
                    detection_mask = detection_mask.unsqueeze(0)                        
                else:
                    XYZ_pred = torch.zeros((3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                    detection_mask = torch.zeros((1, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()
                    plane_XYZ = torch.zeros((1, 3, config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM)).cuda()                        
                    pass


                input_pair.append({'image': images, 'depth': gt_depth, 'mask': gt_masks, 'bbox': gt_boxes, 'extrinsics': extrinsics, 'segmentation': gt_segmentation, 'plane': gt_plane})
                detection_pair.append({'XYZ': XYZ_pred, 'depth': XYZ_pred[1:2], 'mask': detection_mask, 'detection': detections, 'masks': detection_masks, 'feature_map': feature_map[0], 'plane_XYZ': plane_XYZ, 'depth_np': depth_np_pred})

                if 'depth' in options.suffix:
                    ## Apply supervision on reconstructed depthmap (not used currently)
                    if len(detections) > 0:
                        background_mask = torch.clamp(1 - detection_masks.sum(0, keepdim=True), min=0)
                        all_masks = torch.cat([background_mask, detection_masks], dim=0)

                        all_masks = all_masks / all_masks.sum(0, keepdim=True)
                        all_depths = torch.cat([depth_np_pred, plane_XYZ[:, 1]], dim=0)

                        depth_loss = l1LossMask(torch.sum(torch.abs(all_depths[:, 80:560] - gt_depth[:, 80:560]) * all_masks[:, 80:560], dim=0), torch.zeros(config.IMAGE_MIN_DIM, config.IMAGE_MAX_DIM).cuda(), (gt_depth[0, 80:560] > 1e-4).float())
                    else:
                        depth_loss = l1LossMask(depth_np_pred[:, 80:560], gt_depth[:, 80:560], (gt_depth[:, 80:560] > 1e-4).float())
                        pass
                    losses.append(depth_loss)                                                
                    pass

            loss = sum(losses)
            losses = [l.data.item() for l in losses]
            
            epoch_losses.append(losses)
            status = str(epoch + 1) + ' loss: '
            for l in losses:
                status += '%0.5f '%l
                continue


            sys.stdout.write('\r ' + str(sampleIndex) + ' ' + status)
            sys.stdout.flush()
            
            data_iterator.set_description(status)

            loss.backward()
            
            if (sampleIndex + 1) % options.batchSize == 0:
                optimizer.step()
                optimizer.zero_grad()
                pass

            if sampleIndex % 500 < options.batchSize or options.visualizeMode == 'debug':
                ## Visualize intermediate results
                visualizeBatchPair(options, config, input_pair, detection_pair, indexOffset=sampleIndex % 500)
                if (len(detection_pair[0]['detection']) > 0 and len(detection_pair[0]['detection']) < 30) and 'refine' in options.suffix:
                    visualizeBatchRefinement(options, config, input_pair[0], [{'mask': masks_gt, 'plane': planes_gt}, ] + results, indexOffset=sampleIndex % 500, concise=True)
                    pass
                if options.visualizeMode == 'debug' and sampleIndex % 500 >= options.batchSize - 1:
                    exit(1)
                    pass
                pass

            if (sampleIndex + 1) % options.numTrainingImages == 0:
                ## Save models
                print('loss', np.array(epoch_losses).mean(0))
                torch.save(model.state_dict(), options.checkpoint_dir + '/checkpoint.pth')
                torch.save(optimizer.state_dict(), options.checkpoint_dir + '/optim.pth')
                pass
            continue
        continue
    return


if __name__ == '__main__':
    args = parse_args()
    
    args.keyname = 'planercnn'

    args.keyname += '_' + args.anchorType
    if args.dataset != '':
        args.keyname += '_' + args.dataset
        pass
    if args.trainingMode != 'all':
        args.keyname += '_' + args.trainingMode
        pass
    if args.suffix != '':
        args.keyname += '_' + args.suffix
        pass
    
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.test_dir = 'test/' + args.keyname

    if False:
        writeHTML(args.test_dir, ['image_0', 'segmentation_0', 'depth_0', 'depth_0_detection', 'depth_0_detection_ori'], labels=['input', 'segmentation', 'gt', 'before', 'after'], numImages=20, image_width=160, convertToImage=True)
        exit(1)
        
    os.system('rm ' + args.test_dir + '/*.png')
    print('keyname=%s task=%s started'%(args.keyname, args.task))

    train(args)

