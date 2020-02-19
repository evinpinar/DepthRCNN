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

module_path = os.path.abspath(os.path.join('Eigen-pytorch'))
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
from nn_model import Net

def train():
        config = nyu.NYUConfig()
        config.IMAGE_PADDING = False
        path_to_dataset = '../data/NYU_data'

        config.IMAGE_MIN_DIM = 480
        config.IMAGE_MAX_DIM = 640
        batch_size = 8

        dataset_train = nyu.NYUDepthDataset(path_to_dataset, 'train', config)
        dataset_test = nyu.NYUDepthDataset(path_to_dataset, 'test', config)

        print(len(dataset_train), len(dataset_test))
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

        config.STEPS_PER_EPOCH = 350
        config.VALIDATION_STEPS = 50
        config.DEPTH_THRESHOLD = 0
        config.DEPTH_LOSS = True
        config.GRAD_LOSS = False

        model = Net()
        model.cuda()
        trainables_wo_bn = [param for name, param in model.named_parameters() if
                    param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([
            {'params': trainables_wo_bn, 'weight_decay': config.WEIGHT_DECAY},
            {'params': trainables_only_bn}
        ], lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM)

        loss_depth = []
        epochs = 200
        steps = int(len(dataset_train)/batch_size)
        val_steps = int(len(dataset_test)/batch_size)
        loss_depth = []
        val_loss_depth = []

        log_dir = os.path.join('/media/sdb/ornek/checkpoints/', "{}{:%Y%m%dT%H%M}".format('eigen', datetime.datetime.now()))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        checkpoint_path = os.path.join(log_dir, "eigen_{}_*epoch*.pth".format('nyu'))
        checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")

        writer = SummaryWriter(log_dir + '/log/')

        model.train()

        for epoch in range(epochs+1):

                #step = 0
                loss_sum=0
                val_step=0
                val_loss_sum=0
                model.train()
                for i, data in enumerate(train_loader, 0):
                        images = data[0]
                        edges = data[1]
                        gt_depth = data[2]

                        # Wrap in variables
                        images = Variable(images)
                        edges = Variable(edges)
                        gt_depth = Variable(gt_depth)

                        images = images.cuda()
                        edges = edges.cuda()
                        gt_depth = gt_depth.cuda()

                        # Training
                        # input shape is [b,3,240,320]
                        img = F.interpolate(images, size=(240,320), mode='bilinear')
                        pred_depth = model(img)
                        # outpus shape is [1,1,120,160], reshape it back to original size
                        pred_depth=F.interpolate(pred_depth, size=(480,640), mode='bilinear')[:,0]

                        # Compute losses
                        loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                        optimizer.step()
                        optimizer.zero_grad()

                        loss_sum += loss.data.cpu().item() / steps
                        #if step == steps - 1:
                        #    break
                        #step += 1

                model.eval()
                for i, data in enumerate(val_loader, 0):
                    images = data[0]
                    edges = data[1]
                    gt_depth = data[2]

                    # Wrap in variables
                    images = Variable(images)
                    edges = Variable(edges)
                    gt_depth = Variable(gt_depth)

                    images = images.cuda()
                    edges = edges.cuda()
                    gt_depth = gt_depth.cuda()

                    # Training
                    # input shape is [b,3,240,320]
                    img = F.interpolate(images, size=(240,320), mode='bilinear')
                    pred_depth = model(img)
                    # outpus shape is [1,1,120,160], reshape it back to original size
                    pred_depth=F.interpolate(pred_depth, size=(480,640), mode='bilinear')[:,0]

                    # Compute losses
                    val_loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

                    val_loss_sum += val_loss.data.cpu().item() / val_steps
                    #if val_step == val_steps - 1:
                    #    break
                    #val_step += 1

                loss_depth.append(loss_sum)
                val_loss_depth.append(val_loss_sum)

                writer.add_scalar('Train/Loss', loss_sum, epoch)
                writer.add_scalar('Val/Loss', val_loss_sum, epoch)

                if epoch %25 == 0:
                    checkpoint_dir = checkpoint_path.format(epoch)
                    print("Epoch {}/{}.".format(epoch, epochs))
                    print("  Training loss", loss_sum)
                    print("  Validation loss", val_loss_sum)
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

                if epoch == 100:
                    for g in optimizer.param_groups:
                        g['lr'] = config.LEARNING_RATE/10

        writer.close()
        return


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


    model = Net()
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.LEARNING_MOMENTUM)

    epochs = 20
    steps = int(len(dataset_train) / batch_size) # 1402
    val_steps = int(len(dataset_test) / batch_size) # 219
    print("steps: ", steps, "val_steps: ", val_steps)
    loss_depth = []
    val_loss_depth = []

    log_dir = os.path.join('/media/sdb/ornek/checkpoints/', "{}{:%Y%m%dT%H%M}".format('eigen', datetime.datetime.now()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    checkpoint_path = os.path.join(log_dir, "eigen_{}_*epoch*.pth".format('nyu'))
    checkpoint_path = checkpoint_path.replace("*epoch*", "{:04d}")

    writer = SummaryWriter(log_dir + '/log/')

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

            images = Variable(images)
            gt_depth = Variable(gt_depth)

            images = images.cuda()
            gt_depth = gt_depth.cuda()

            # print("img shape: ", images.shape, " depth shape: ", gt_depth.shape)

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')

            pred_depth = model(img)
            # outpus shape is [1,1,120,160], reshape it back to original size
            pred_depth = F.interpolate(pred_depth, size=(480, 640), mode='bilinear')[:, 0]

            # Compute losses
            loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 0).float())

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

                images = Variable(images)
                gt_depth = Variable(gt_depth)

                images = images.cuda()
                gt_depth = gt_depth.cuda()

                # Training
                # input shape is [b,3,240,320]
                img = F.interpolate(images, size=(240, 320), mode='bilinear')
                pred_depth = model(img)
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

        if epoch == 100:
            for g in optimizer.param_groups:
                g['lr'] = config.LEARNING_RATE / 10

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

    model = Net()
    model.cuda()

    checkpoint_dir = '../data/checkpoints/eigen20200129T0152/eigen_nyu_0020.pth'
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    errors = []
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(test_loader, 0):
            images = inputs[0]
            gt_depth = inputs[1]

            images = Variable(images)
            gt_depth = Variable(gt_depth)

            images = images.cuda()
            gt_depth = gt_depth.cuda()

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')
            pred_depth = model(img)
            # outpus shape is [1,1,120,160], reshape it back to original size
            pred_depth = F.interpolate(pred_depth, size=(480, 640), mode='bilinear')[:, 0]

            depth_pred = pred_depth[0].detach().cpu().numpy()
            depth_gt = gt_depth[0].detach().cpu().numpy()

            err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
            errors.append(err)

            if i%50 == 0:
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

    print(rel_sqr)

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

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    log_10 = np.mean(np.abs(np.log10(gt) - np.log10(pred)))

    return abs_rel, log_10, rmse, rmse_log, a1, a2, a3


def evalScannet2():
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

    model = Net()
    model.cuda()

    checkpoint_dir = '../data/checkpoints/eigen20200118T1942/eigen_nyu_0020.pth'
    checkpoint = torch.load(checkpoint_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    num_samples = len(dataset_test)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    log_10 = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    model.eval()
    with torch.no_grad():
        for i, inputs in enumerate(test_loader, 0):
            images = inputs[0]
            gt_depth = inputs[1]

            images = Variable(images)
            gt_depth = Variable(gt_depth)

            images = images.cuda()
            gt_depth = gt_depth.cuda()

            # Training
            # input shape is [b,3,240,320]
            img = F.interpolate(images, size=(240, 320), mode='bilinear')
            pred_depth = model(img)
            # outpus shape is [1,1,120,160], reshape it back to original size
            pred_depth = F.interpolate(pred_depth, size=(480, 640), mode='bilinear')[:, 0]

            depth_pred = pred_depth[0].detach().cpu().numpy()
            depth_gt = gt_depth[0].detach().cpu().numpy()

            #err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
            #errors.append(err)

            mask = (depth_gt > 0)
            abs_rel[i], log_10[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(depth_gt[mask],
                                                                                              depth_pred[mask])
            if np.isnan(rmse_log[i]):
                rmse_log[i] = 0

            if i%50 == 0:
                print(i)

    #e = np.array(errors).mean(0).tolist()
    e = [abs_rel.sum()/i,log_10.sum()/i,rmse.sum()/i,rmse_log.sum()/i,a1.sum()/i,a2.sum()/i,a3.sum()/i]

    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('absrel', 'log_10', 'rmse', 'rmselog',
                                        'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6]))



if __name__ == '__main__':
    evalScannet()
