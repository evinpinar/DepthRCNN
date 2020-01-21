
import os
import sys
import datetime
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

module_path = os.path.abspath(os.path.join('../DepthRCNN'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('ldid'))
if module_path not in sys.path:
    sys.path.append(module_path)

torch.cuda.set_device(0)

from config import Config
import nyu
import scannet
import utils
import visualize
from visualize import display_images
import models.model as modellib
from models.model import log

from models.model_maskdepthrcnn import *

import model_utils
from model import *
from focal import *
from grad_loss import *
from depth_loss import *



def trainScannet():

    path_to_dataset = "../data/SCANNET/"
    config = scannet.ScannetConfig()
    config.IMAGE_PADDING = False

    config.IMAGE_MIN_DIM = 480
    config.IMAGE_MAX_DIM = 640
    batch_size = 30

    dataset_train = scannet.ScannetDepthDataset(subset='train', config=config, scannet_data=path_to_dataset)
    dataset_test = scannet.ScannetDepthDataset(subset='val', config=config, scannet_data=path_to_dataset)

    print(len(dataset_train), len(dataset_test))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1,
                                             drop_last=True)


    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.DEPTH_THRESHOLD = 0
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    config.USE_MINI_MASK = False
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU

    model_path = 'ldid/models/model.pth'
    model = Model(ResidualBlock, UpProj_Block, 1)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model.cuda()

    sem_params = list(model.sem_up_conv1.parameters()) + list(model.sem_up_conv2.parameters()) + list(
        model.sem_up_conv3.parameters()) + list(model.sem_up_conv4.parameters()) + list(
        model.sem_skip_up1.parameters()) + list(model.sem_skip_up2.parameters()) + list(
        model.sem_skip_up3.parameters()) + list(model.sem_conv3.parameters())
    dep_params = list(model.dep_up_conv1.parameters()) + list(model.dep_up_conv2.parameters()) + list(
        model.dep_up_conv3.parameters()) + list(model.dep_up_conv4.parameters()) + list(
        model.dep_skip_up1.parameters()) + list(model.dep_skip_up2.parameters()) + list(
        model.dep_skip_up3.parameters()) + list(model.dep_conv3.parameters()) + list(model.upsample.parameters())
    lsu_params = list(model.LS_d11.parameters()) + list(model.LS_d12.parameters()) + list(
        model.LS_d21.parameters()) + list(model.LS_d22.parameters()) + list(model.LS_d31.parameters()) + list(
        model.LS_d32.parameters()) + list(model.LS_s11.parameters()) + list(model.LS_s12.parameters()) + list(
        model.LS_s21.parameters()) + list(model.LS_s22.parameters()) + list(model.LS_s31.parameters()) + list(
        model.LS_s32.parameters())

    ### Find the backbone parameters
    for param in sem_params:
        param.requires_grad = False

    for param in lsu_params:
        param.requires_grad = False

    for param in dep_params:
        param.requires_grad = False

    backbone_params = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backbone_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": sem_params, "lr": 1e-3, "name": 'sem'},
            {"params": dep_params, "lr": 1e-3, "name": 'dep'},
            {"params": lsu_params, "lr": 1e-3, "name": 'lsu'},
            {"params": backbone_params, "lr": 1e-3, "name": 'back'}
        ],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-08,
    )

    focal_loss = FocalLoss(alpha=0.75, reduction='mean')

    epochs = 20
    steps = int(len(dataset_train) / batch_size) # 1402
    val_steps = int(len(dataset_test) / batch_size) # 219
    print("steps: ", steps, "val_steps: ", val_steps)
    loss_depth = []
    val_loss_depth = []

    log_dir = os.path.join('/media/sdb/ornek/checkpoints/', "{}{:%Y%m%dT%H%M}".format('jiao', datetime.datetime.now()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_path = os.path.join(log_dir, "jiao_{}_*epoch*.pth".format('nyu'))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")

    writer = SummaryWriter(log_dir + '/log/')

    ### initially set depth and backbone params trainable

    for param in dep_params:
        param.requires_grad = True

    for param in backbone_params:
        param.requires_grad = False

    model.train()

    for epoch in range(epochs + 1):

        step = 0
        loss_sum = 0
        val_step = 0
        val_loss_sum = 0
        model.train()
        for i, inputs in enumerate(train_loader, 0):

            images = inputs[0]
            gt_depth = inputs[1]
            gt_labels = inputs[2]

            images = Variable(images)
            gt_depth = Variable(gt_depth)
            gt_labels = Variable(gt_labels)

            images = images.cuda()
            gt_depth = gt_depth.cuda()
            gt_labels = gt_labels.cuda()

            # print("img shape: ", images.shape, " depth shape: ", gt_depth.shape)

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')

            pred_depth, pred_labels = model(img)
            # outpus shape is [1,1,120,160], reshape it back to original size
            #pred_depth = F.interpolate(pred_depth, size=(480, 640), mode='bilinear')[:, 0]

            # L1 depth aware loss
            depth_loss = depth_loss(pred_depth, gt_depth)

            # focal loss
            seg_loss = focal_loss(pred_labels, gt_labels.long())

            # gradient losses for depth and segmentation
            grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth)
            depth_grad = grad_loss(grad_fake, grad_real)

            _, preds_lbl = pred_labels.max(1) # [B, C, H, W] to [B, H, W]
            grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
            segm_grad = grad_loss(grad_fake2, grad_real2)


            loss = depth_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.data.cpu().item() / steps
            # if step == steps - 1:
            #    break
            # step += 1

        model.eval()
        with torch.no_grad():
            for i, inputs in enumerate(val_loader, 0):
                images = inputs[0]
                gt_depth = inputs[1]
                gt_labels = inputs[2]

                images = Variable(images)
                gt_depth = Variable(gt_depth)
                gt_labels = Variable(gt_labels)

                images = images.cuda()
                gt_depth = gt_depth.cuda()
                gt_labels = gt_labels.cuda()

                # Training
                # input shape is [b,3,240,320]
                img = F.interpolate(images, size=(240, 320), mode='bilinear')
                pred_depth, pred_labels = model(img)
                # outpus shape is [1,1,120,160], reshape it back to original size
                pred_depth = F.interpolate(pred_depth, size=(480, 640), mode='bilinear')[:, 0]

                # Compute losses
                val_loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

                val_loss_sum += val_loss.data.cpu().item() / val_steps
                # if val_step == val_steps - 1:
                #    break
                # val_step += 1

        loss_depth.append(loss_sum)
        val_loss_depth.append(val_loss_sum)

        writer.add_scalar('Train/Loss', loss_sum, epoch)
        writer.add_scalar('Val/Loss', val_loss_sum, epoch)

        print("  Training loss", loss_sum)
        print("  Validation loss", val_loss_sum)

        if epoch % 5 == 0:
            checkpoint_dir = checkpoint_path.format(epoch)
            print("Epoch {}/{}.".format(epoch, epochs))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_sum,
            }, checkpoint_dir)

        loss_np = np.array(loss_depth)
        val_loss_np = np.array(val_loss_depth)
        plt.figure("loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "loss.png")
        plt.savefig(save_path)

        if epoch == 4:
            # unfreeze semantic sem and lsu, freeze depth
            for param in sem_params:
                param.requires_grad = True

            for param in lsu_params:
                param.requires_grad = True

            for param in dep_params:
                param.requires_grad = False

            for g in optimizer.param_groups:
                if g['name'] == 'back':
                    g['lr'] = 1e-5

        if epoch == 9:
            # learn all weights (unfreeze dep)
            for param in dep_params:
                param.requires_grad = True

            for g in optimizer.param_groups:
                if g['name'] == 'back':
                    g['lr'] = 1e-2
                else:
                    g['lr'] = 1e-4

        if epoch == 20:
            for g in optimizer.param_groups:
                g['lr'] = config.LEARNING_RATE / 10

    writer.close()
    return

