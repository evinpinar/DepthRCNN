

import numpy as np
import torch
# import pandas as pd
import os
import cv2
from collections import Counter
import pickle


# Error computation based on https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py
# Also check https://github.com/ialhashim/DenseDepth/blob/master/utils.py
def compute_errors(gt, pred):

    ## It does not mask out the irrelevant values from ground truth depth! (<10e4)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def CalculateLosses(pred, gt):

    pred = pred.reshape(-1)
    gt = gt.reshape(-1)

    # Filtering, similar to the one used in official implementation
    mask = (gt > 1e-3) & (gt < 80)
    masked_pred = torch.masked_select(pred, mask)
    masked_gt = torch.masked_select(gt, mask)

    masked_pred[masked_pred < 1e-3] = 1e-3
    masked_pred[masked_pred > 80] = 80

    rmse = torch.sqrt(torch.mean((masked_gt-masked_pred)**2)).item()
    rmse_log = torch.sqrt(((torch.log(masked_gt) - torch.log(masked_pred))**2).mean()).item()
    abs_rel = torch.mean(torch.abs(masked_gt-masked_pred) / masked_gt).item()
    sq_rel = torch.mean((masked_gt-masked_pred)**2 / masked_gt).item()

    return [rmse, rmse_log, abs_rel, sq_rel]





#print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
#print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))