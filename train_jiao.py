
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
    batch_size = 12

    dataset_train = scannet.ScannetDepthDataset(subset='train', config=config, scannet_data=path_to_dataset)
    dataset_test = scannet.ScannetDepthDataset(subset='val', config=config, scannet_data=path_to_dataset)

    print(len(dataset_train), len(dataset_test))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0,
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

    model = Model(ResidualBlock, UpProj_Block, batch_size)


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

    back_names = []
    backbone_params = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backbone_params.append(param)
            back_names.append(name)

    # Load only backbone parameters
    model_path = '../data/jiao_model.pth'
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_state_dict = {k: v for k, v in state_dict.items() if k in back_names}
    state = model.state_dict()
    state.update(new_state_dict)
    model.load_state_dict(state)


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
    loss_segm = []
    loss_grad1 = []
    loss_grad2 = []
    loss_total = []
    val_loss_depth = []
    val_loss_segm = []
    val_loss_grad1 = []
    val_loss_grad2 = []
    val_loss_total = []


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
        param.requires_grad = True

    model.train()

    for epoch in range(epochs + 1):

        loss_sum = 0
        loss_depth_sum = 0
        loss_segm_sum = 0
        loss_grad1_sum = 0
        loss_grad2_sum = 0
        val_loss_sum = 0
        val_loss_depth_sum = 0
        val_loss_segm_sum = 0
        val_loss_grad1_sum = 0
        val_loss_grad2_sum = 0

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

            #print("img shape: ", images.shape, " depth shape: ", gt_depth.shape, " gt labels: ", gt_labels.shape)

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')

            pred_depth, pred_labels = model(img)
            # outpus shape is [b,1,480,640], reshape it back to original size

            #print("pred shape ", pred_depth.shape, pred_labels.shape)
            pred_depth = pred_depth[:, 0]
            #print(" pred after: ", pred_depth.shape)

            # L1 depth aware loss
            dep_loss = depth_loss(pred_depth, gt_depth)

            # focal loss
            seg_loss = focal_loss(pred_labels, gt_labels.long())

            # gradient losses for depth and segmentation
            grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth.unsqueeze(1))
            depth_grad = grad_loss(grad_fake, grad_real)

            # compare predicted depth grad with segmentation ground truth grad
            _, preds_lbl = pred_labels.max(1) # [B, C, H, W] to [B, H, W]
            #grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
            #grad_real2 = imgrad_yx(gt_labels.unsqueeze(1))
            grad_fake2 = imgrad_yx(preds_lbl.float().unsqueeze(1))
            segm_grad = grad_loss(grad_fake, grad_fake2)

            #print("depth: ", dep_loss, "seg: ", seg_loss, "grad1: ", depth_grad, "grad2: ", segm_grad)

            loss = dep_loss + seg_loss + depth_grad + segm_grad

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.data.cpu().item() / steps
            loss_depth_sum += dep_loss.data.cpu().item() / steps
            loss_segm_sum += seg_loss.data.cpu().item() / steps
            loss_grad1_sum += depth_grad.data.cpu().item() / steps
            loss_grad2_sum += segm_grad.data.cpu().item() / steps


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
                # outpus shape is [b,1,480,640], reshape it back to original size
                pred_depth = pred_depth[:, 0]


                # L1 depth aware loss
                dep_loss = depth_loss(pred_depth, gt_depth)

                # focal loss
                seg_loss = focal_loss(pred_labels, gt_labels.long())

                # gradient losses for depth and segmentation
                grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth.unsqueeze(1))
                depth_grad = grad_loss(grad_fake, grad_real)

                # compare predicted depth grad with segmentation ground truth grad
                _, preds_lbl = pred_labels.max(1)  # [B, C, H, W] to [B, H, W]
                # grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
                #grad_real2 = imgrad_yx(gt_labels.unsqueeze(1))
                grad_fake2 = imgrad_yx(preds_lbl.float().unsqueeze(1))
                segm_grad = grad_loss(grad_fake, grad_fake2)

                val_loss = dep_loss + seg_loss + depth_grad + segm_grad

                val_loss_sum += val_loss.data.cpu().item() / val_steps
                val_loss_depth_sum += dep_loss.data.cpu().item() / val_steps
                val_loss_segm_sum += seg_loss.data.cpu().item() / val_steps
                val_loss_grad1_sum += depth_grad.data.cpu().item() / val_steps
                val_loss_grad2_sum += segm_grad.data.cpu().item() / val_steps
                # if val_step == val_steps - 1:
                #    break
                # val_step += 1

        loss_depth.append(loss_depth_sum)
        loss_segm.append(loss_segm_sum)
        loss_grad1.append(loss_grad1_sum)
        loss_grad2.append(loss_grad2_sum)
        loss_total.append(loss_sum)

        val_loss_depth.append(val_loss_depth_sum)
        val_loss_segm.append(val_loss_segm_sum)
        val_loss_grad1.append(val_loss_grad1_sum)
        val_loss_grad2.append(val_loss_grad2_sum)
        val_loss_total.append(val_loss_sum)

        writer.add_scalar('Train/Loss', loss_sum, epoch)
        writer.add_scalar('Val/Loss', val_loss_sum, epoch)

        print(" epoch: ", epoch)
        print("  Training loss", loss_sum)
        print("  Validation loss", val_loss_sum)

        if epoch % 2 == 0:
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
        plt.figure(" depth loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "depth_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_segm)
        val_loss_np = np.array(val_loss_segm)
        plt.figure("segm loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "segm_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_grad1)
        val_loss_np = np.array(val_loss_grad1)
        plt.figure("depth grad loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "depth_grad_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_grad2)
        val_loss_np = np.array(val_loss_grad2)
        plt.figure("segm grad loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "segm_grad_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_total)
        val_loss_np = np.array(val_loss_total)
        plt.figure("total loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "total_loss.png")
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


def trainScannet_1cont():

    path_to_dataset = "../data/SCANNET/"
    config = scannet.ScannetConfig()
    config.IMAGE_PADDING = False

    config.IMAGE_MIN_DIM = 480
    config.IMAGE_MAX_DIM = 640
    batch_size = 10

    dataset_train = scannet.ScannetDepthDataset(subset='train', config=config, scannet_data=path_to_dataset)
    dataset_test = scannet.ScannetDepthDataset(subset='val', config=config, scannet_data=path_to_dataset)

    print(len(dataset_train), len(dataset_test))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0,
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

    model = Model(ResidualBlock, UpProj_Block, batch_size)
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

    back_names = []
    backbone_params = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backbone_params.append(param)
            back_names.append(name)

    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    optimizer = torch.optim.Adam(
        [
            {"params": sem_params, "lr": 1e-2, "name": 'sem'},
            {"params": dep_params, "lr": 1e-2, "name": 'dep'},
            {"params": lsu_params, "lr": 1e-2, "name": 'lsu'},
            {"params": backbone_params, "lr": 1e-4, "name": 'back'}
        ],
        lr=1e-4, betas=(0.9, 0.999), eps=1e-08,
    )

    for param in sem_params:
        param.requires_grad = True

    for param in lsu_params:
        param.requires_grad = True

    for param in dep_params:
        param.requires_grad = True

    for param in backbone_params:
        param.requires_grad = False

    checkpoint_dir = '../data/checkpoints/jiao20200123T1149/jiao_nyu_0020.pth'
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("model loaded")
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("optim loaded")

    focal_loss = FocalLoss(alpha=0.75, reduction='mean')

    epochs = 20
    steps = int(len(dataset_train) / batch_size) # 1402
    val_steps = int(len(dataset_test) / batch_size) # 219
    print("steps: ", steps, "val_steps: ", val_steps)
    loss_depth = []
    loss_segm = []
    loss_grad1 = []
    loss_grad2 = []
    loss_total = []
    val_loss_depth = []
    val_loss_segm = []
    val_loss_grad1 = []
    val_loss_grad2 = []
    val_loss_total = []


    log_dir = os.path.join('/media/sdb/ornek/checkpoints/', "{}{:%Y%m%dT%H%M}".format('jiao', datetime.datetime.now()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_path = os.path.join(log_dir, "jiao_{}_*epoch*.pth".format('nyu'))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")

    writer = SummaryWriter(log_dir + '/log/')

    ### initially set depth and backbone params trainable


    model.train()

    for epoch in range(epochs + 1):

        loss_sum = 0
        loss_depth_sum = 0
        loss_segm_sum = 0
        loss_grad1_sum = 0
        loss_grad2_sum = 0
        val_loss_sum = 0
        val_loss_depth_sum = 0
        val_loss_segm_sum = 0
        val_loss_grad1_sum = 0
        val_loss_grad2_sum = 0

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

            #print("img shape: ", images.shape, " depth shape: ", gt_depth.shape, " gt labels: ", gt_labels.shape)

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')

            pred_depth, pred_labels = model(img)
            # outpus shape is [b,1,480,640], reshape it back to original size

            #print("pred shape ", pred_depth.shape, pred_labels.shape)
            pred_depth = pred_depth[:, 0]
            #print(" pred after: ", pred_depth.shape)

            # L1 depth aware loss
            dep_loss = depth_loss(pred_depth, gt_depth)

            # focal loss
            seg_loss = focal_loss(pred_labels, gt_labels.long())

            # gradient losses for depth and segmentation
            grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth.unsqueeze(1))
            depth_grad = grad_loss(grad_fake, grad_real)

            # compare predicted depth grad with segmentation ground truth grad
            _, preds_lbl = pred_labels.max(1) # [B, C, H, W] to [B, H, W]
            #grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
            #grad_real2 = imgrad_yx(gt_labels.unsqueeze(1))
            grad_fake2 = imgrad_yx(preds_lbl.float().unsqueeze(1))
            segm_grad = grad_loss(grad_fake, grad_fake2)

            #print("depth: ", dep_loss, "seg: ", seg_loss, "grad1: ", depth_grad, "grad2: ", segm_grad)

            loss = dep_loss + seg_loss + depth_grad + segm_grad

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.data.cpu().item() / steps
            loss_depth_sum += dep_loss.data.cpu().item() / steps
            loss_segm_sum += seg_loss.data.cpu().item() / steps
            loss_grad1_sum += depth_grad.data.cpu().item() / steps
            loss_grad2_sum += segm_grad.data.cpu().item() / steps


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
                # outpus shape is [b,1,480,640], reshape it back to original size
                pred_depth = pred_depth[:, 0]


                # L1 depth aware loss
                dep_loss = depth_loss(pred_depth, gt_depth)

                # focal loss
                seg_loss = focal_loss(pred_labels, gt_labels.long())

                # gradient losses for depth and segmentation
                grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth.unsqueeze(1))
                depth_grad = grad_loss(grad_fake, grad_real)

                # compare predicted depth grad with segmentation ground truth grad
                _, preds_lbl = pred_labels.max(1)  # [B, C, H, W] to [B, H, W]
                # grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
                #grad_real2 = imgrad_yx(gt_labels.unsqueeze(1))
                grad_fake2 = imgrad_yx(preds_lbl.float().unsqueeze(1))
                segm_grad = grad_loss(grad_fake, grad_fake2)

                val_loss = dep_loss + seg_loss + depth_grad + segm_grad

                val_loss_sum += val_loss.data.cpu().item() / val_steps
                val_loss_depth_sum += dep_loss.data.cpu().item() / val_steps
                val_loss_segm_sum += seg_loss.data.cpu().item() / val_steps
                val_loss_grad1_sum += depth_grad.data.cpu().item() / val_steps
                val_loss_grad2_sum += segm_grad.data.cpu().item() / val_steps
                # if val_step == val_steps - 1:
                #    break
                # val_step += 1

        loss_depth.append(loss_depth_sum)
        loss_segm.append(loss_segm_sum)
        loss_grad1.append(loss_grad1_sum)
        loss_grad2.append(loss_grad2_sum)
        loss_total.append(loss_sum)

        val_loss_depth.append(val_loss_depth_sum)
        val_loss_segm.append(val_loss_segm_sum)
        val_loss_grad1.append(val_loss_grad1_sum)
        val_loss_grad2.append(val_loss_grad2_sum)
        val_loss_total.append(val_loss_sum)

        writer.add_scalar('Train/Loss', loss_sum, epoch)
        writer.add_scalar('Val/Loss', val_loss_sum, epoch)

        print(" epoch: ", epoch)
        print("  Training loss", loss_sum)
        print("  Validation loss", val_loss_sum)

        if epoch % 2 == 0:
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
        plt.figure(" depth loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "depth_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_segm)
        val_loss_np = np.array(val_loss_segm)
        plt.figure("segm loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "segm_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_grad1)
        val_loss_np = np.array(val_loss_grad1)
        plt.figure("depth grad loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "depth_grad_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_grad2)
        val_loss_np = np.array(val_loss_grad2)
        plt.figure("segm grad loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "segm_grad_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_total)
        val_loss_np = np.array(val_loss_total)
        plt.figure("total loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "total_loss.png")
        plt.savefig(save_path)

    writer.close()
    return



def trainScannet2():

    path_to_dataset = "../data/SCANNET/"
    config = scannet.ScannetConfig()
    config.IMAGE_PADDING = False

    config.IMAGE_MIN_DIM = 480
    config.IMAGE_MAX_DIM = 640
    batch_size = 12

    dataset_train = scannet.ScannetDepthDataset(subset='train', config=config, scannet_data=path_to_dataset)
    dataset_test = scannet.ScannetDepthDataset(subset='val', config=config, scannet_data=path_to_dataset)

    print(len(dataset_train), len(dataset_test))
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0,
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

    model = Model(ResidualBlock, UpProj_Block, batch_size)


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

    back_names = []
    backbone_params = list()
    for name, param in model.named_parameters():
        if param.requires_grad:
            backbone_params.append(param)
            back_names.append(name)

    # Load only backbone parameters
    model_path = '../data/jiao_model.pth'
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    new_state_dict = {k: v for k, v in state_dict.items() if k in back_names}
    state = model.state_dict()
    state.update(new_state_dict)
    model.load_state_dict(state)

    for name, param in model.named_parameters():
        param.requires_grad = True

    trainables_wo_bn = [param for name, param in model.named_parameters() if
                        param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
    optimizer = optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': config.WEIGHT_DECAY},
        {'params': trainables_only_bn}
    ], lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM)

    epochs = 20
    steps = int(len(dataset_train) / batch_size) # 1402
    val_steps = int(len(dataset_test) / batch_size) # 219
    print("steps: ", steps, "val_steps: ", val_steps)
    loss_depth = []
    loss_segm = []
    loss_total = []
    val_loss_depth = []
    val_loss_segm = []
    val_loss_total = []


    log_dir = os.path.join('/media/sdb/ornek/checkpoints/', "{}{:%Y%m%dT%H%M}".format('jiao', datetime.datetime.now()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_path = os.path.join(log_dir, "jiao_{}_*epoch*.pth".format('nyu'))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")

    writer = SummaryWriter(log_dir + '/log/')

    ### initially set depth and backbone params trainable


    focal_loss = FocalLoss(alpha=0.75, reduction='mean')

    model.train()

    for epoch in range(epochs + 1):

        loss_sum = 0
        loss_depth_sum = 0
        loss_segm_sum = 0
        val_loss_sum = 0
        val_loss_depth_sum = 0
        val_loss_segm_sum = 0

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

            #print("img shape: ", images.shape, " depth shape: ", gt_depth.shape, " gt labels: ", gt_labels.shape)

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')

            pred_depth, pred_labels = model(img)
            # outpus shape is [b,1,480,640], reshape it back to original size

            #print("pred shape ", pred_depth.shape, pred_labels.shape)
            pred_depth = pred_depth[:, 0]
            #print(" pred after: ", pred_depth.shape)

            # L1 depth aware loss
            dep_loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

            # focal loss
            seg_loss = focal_loss(pred_labels, gt_labels.long())

            # gradient losses for depth and segmentation
            # grad_real, grad_fake = imgrad_yx(gt_depth.unsqueeze(1)), imgrad_yx(pred_depth.unsqueeze(1))
            # depth_grad = grad_loss(grad_fake, grad_real)

            # compare predicted depth grad with segmentation ground truth grad
            #_, preds_lbl = pred_labels.max(1) # [B, C, H, W] to [B, H, W]
            #grad_real2, grad_fake2 = imgrad_yx(gt_labels.unsqueeze(1)), imgrad_yx(preds_lbl.float().unsqueeze(1))
            #grad_real2 = imgrad_yx(gt_labels.unsqueeze(1))
            #grad_fake2 = imgrad_yx(preds_lbl.float().unsqueeze(1))
            #segm_grad = grad_loss(grad_fake, grad_fake2)

            #print("depth: ", dep_loss, "seg: ", seg_loss, "grad1: ", depth_grad, "grad2: ", segm_grad)

            loss = dep_loss + seg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()
            optimizer.zero_grad()

            loss_sum += loss.data.cpu().item() / steps
            loss_depth_sum += dep_loss.data.cpu().item() / steps
            loss_segm_sum += seg_loss.data.cpu().item() / steps


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
                # outpus shape is [b,1,480,640], reshape it back to original size
                pred_depth = pred_depth[:, 0]

                # L1 depth aware loss
                dep_loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

                # focal loss
                seg_loss = focal_loss(pred_labels, gt_labels.long())

                val_loss = dep_loss + seg_loss

                val_loss_sum += val_loss.data.cpu().item() / val_steps
                val_loss_depth_sum += dep_loss.data.cpu().item() / val_steps
                val_loss_segm_sum += seg_loss.data.cpu().item() / val_steps
                # if val_step == val_steps - 1:
                #    break
                # val_step += 1

        loss_depth.append(loss_depth_sum)
        loss_segm.append(loss_segm_sum)
        loss_total.append(loss_sum)

        val_loss_depth.append(val_loss_depth_sum)
        val_loss_segm.append(val_loss_segm_sum)
        val_loss_total.append(val_loss_sum)

        writer.add_scalar('Train/Loss', loss_sum, epoch)
        writer.add_scalar('Val/Loss', val_loss_sum, epoch)

        print(" epoch: ", epoch)
        print("  Training loss", loss_sum)
        print("  Validation loss", val_loss_sum)

        if epoch % 2 == 0:
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
        plt.figure(" depth loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "depth_loss.png")
        plt.savefig(save_path)

        loss_np = np.array(loss_segm)
        val_loss_np = np.array(val_loss_segm)
        plt.figure("segm loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "segm_loss.png")
        plt.savefig(save_path)


        loss_np = np.array(loss_total)
        val_loss_np = np.array(val_loss_total)
        plt.figure("total loss")
        plt.gcf().clear()
        plt.plot(loss_np, label='train')
        plt.plot(val_loss_np, label='valid')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        save_path = os.path.join(log_dir, "total_loss.png")
        plt.savefig(save_path)


    writer.close()
    return



def evalScannet():

    path_to_dataset = "../data/SCANNET/"
    config = scannet.ScannetConfig()
    config.IMAGE_PADDING = False

    config.IMAGE_MIN_DIM = 480
    config.IMAGE_MAX_DIM = 640
    batch_size = 1

    dataset_test = scannet.ScannetDepthDataset(subset='test', config=config, scannet_data=path_to_dataset)

    print(len(dataset_test))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1,
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

    model = Model(ResidualBlock, UpProj_Block, batch_size)

    model.cuda()

    checkpoint_dir = '../data/checkpoints/jiao20200127T1811/jiao_nyu_0020.pth'
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    errors = []
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(test_loader, 0):
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
            # outpus shape is [b,1,480,640], reshape it back to original size
            pred_depth = pred_depth[:, 0]

            depth_pred = pred_depth[0].detach().cpu().numpy()
            depth_gt = gt_depth[0].detach().cpu().numpy()

            err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
            errors.append(err)

            if i % 50 == 0:
                print(i, err)

    e = np.array(errors).mean(0).tolist()
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))


def evaluateDepths(predDepths, gtDepths, printInfo=False):
    """Evaluate depth reconstruction accuracy"""

    masks = gtDepths > 1e-4

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)

    rmse_log = np.sqrt(
        (pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)

    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels

    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels

    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
                1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass

    rel = round(rel, 5)
    rel_sqr = round(rel_sqr, 5)
    log10 = round(log10, 5)
    rmse = round(rmse, 5)
    rmse_log = round(rmse_log, 5)
    accuracy_1 = round(accuracy_1, 5)
    accuracy_2 = round(accuracy_2, 5)
    accuracy_3 = round(accuracy_3, 5)

    return [rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3]


if __name__ == '__main__':
    evalScannet()
