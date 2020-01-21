
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from imageio import imread, imsave
except:
    from scipy.misc import imread, imsave

from skimage import feature
from scipy import ndimage
import time

import os
import scipy.io as io
import h5py
import cv2


def compute_depth_metrics(input, target, mask=None):
    if mask is None:
        rmse = np.sqrt(np.mean((input - target) ** 2))
        rmse_log = np.sqrt(np.mean(np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2)

        avg_log10 = np.mean(
            np.abs(
                np.log10(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log10(np.clip(target, a_min=1e-12, a_max=1e12))))

        rel = np.mean(np.abs(input - target) / target)
    else:
        N = np.sum(mask)

        diff = mask * (input - target)
        diff = diff ** 2
        diff_log = mask * (np.log(np.clip(input, a_min=1e-12, a_max=1e12)) - np.log(
            np.clip(target, a_min=1e-12, a_max=1e12))) ** 2
        mse = np.sum(diff)
        mse_log = np.sum(diff_log)
        rmse = np.sqrt(float(mse) / N)
        rmse_log = np.sqrt(float(mse_log) / N)

        avg_log10 = np.sum(
            mask * np.abs(np.log10(np.clip(input, a_min=1e-12, a_max=1e8))
                          - np.log10(np.clip(target, a_min=1e-8, a_max=1e8))))
        avg_log10 = float(avg_log10) / N

        rel = float(np.sum(np.abs(input - target) / target)) / N

    acc_map = np.max((target / (input + 1e-8), input / (target + 1e-8)), axis=0)
    acc_1_map = acc_map < 1.25
    acc_2_map = acc_map < 1.25 ** 2
    acc_3_map = acc_map < 1.25 ** 3
    if mask is not None:
        acc_1_map[mask == 0] = False
        acc_2_map[mask == 0] = False
        acc_3_map[mask == 0] = False

        N = np.sum(mask)
    else:
        N = np.prod(input.shape)

    acc_1 = len(acc_1_map[acc_1_map == True]) / N
    acc_2 = len(acc_2_map[acc_2_map == True]) / N
    acc_3 = len(acc_2_map[acc_3_map == True]) / N

    return rel, 0, avg_log10, rmse, rmse_log, acc_1, acc_2, acc_3



def compute_depth_boundary_error(edges_gt, pred, mask=None, low_thresh=0.15, high_thresh=0.3):
    # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:

        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)

        # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=low_thresh,
                                  high_threshold=high_thresh)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood

        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges
        if mask is None:
            mask = np.ones(shape=E_fin_est_filt.shape)
        E_fin_est_filt = E_fin_est_filt * mask
        D_gt = D_gt * mask

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com, edges_est, D_est