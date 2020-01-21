
import scannet
import nyu
from sun import SunConfig
from models.model_maskdepthrcnn import *
from timeit import default_timer as timer
from config import Config
import utils
from tensorboardX import SummaryWriter
import imgaug.augmenters as iaa

from evaluate_utils import *
import logging
from sharpnet_eval import *
import utils_original


def evaluate_solodepth_scannet():

    print("Model evaluation on NYU occlusion boundary dataset!")

    dataset_path = '../NYU_occlusion_boundaries/'
    indices = np.load(dataset_path + 'indices.npy')
    print("Size of NYU occlusion dataset: ", len(indices))

    '''
    config = scannet.ScannetConfig()

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.DEPTH_THRESHOLD = 0
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    config.USE_MINI_MASK = False
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28
    '''

    config = nyu.NYUConfig()
    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.CHAM_LOSS = False
    config.GRAD_LOSS = False
    config.CHAM_COMBINE = False

    depth_model = DepthCNN(config)
    depth_model.cuda()

    checkpoint_dir = 'checkpoints/scannet20191218T1651/mask_rcnn_scannet_0020.pth'
    #checkpoint_dir = 'checkpoints/nyudepth20191230T1055/mask_rcnn_nyudepth_0200.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    errors = []
    errors_sharp = []
    errors_occ = []

    for i in indices:

        depth = np.load(dataset_path + str(i) + '_depth.npy')/4
        rgb = imread(dataset_path + str(i) + '_img.png')
        occ_bound = imread(dataset_path + str(i) + '_ob.png')

        rgb, window, scale, padding = utils_original.resize_image(
            np.array(rgb),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        depth, _, _, _ = utils_original.resize_depth(
            np.array(depth),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)

        occ_bound, _, _, _ = utils_original.resize_depth(
            np.array(occ_bound),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)

        #rgb = mold_image(rgb, config)
        rgb = rgb.transpose(2,0,1)

        gt_depth = torch.tensor(depth).unsqueeze(0)
        rgb = torch.tensor(rgb).unsqueeze(0)

        images = Variable(rgb)
        gt_depth = Variable(gt_depth)

        images = images.cuda().float()
        gt_depth = gt_depth.cuda().float()
        # print("image input shape: ", images.shape)

        depth_np = depth_model.predict([images, gt_depth], mode='inference')

        depth_pred = depth_np[0]
        # print('pred_depth_shape: ', depth_pred.shape)
        depth_pred = depth_pred[0, 80:560, :].detach().cpu().numpy()
        depth_gt = gt_depth[0, 80:560, :].cpu().numpy()
        #print("pred: ", depth_pred[:5, :5])
        #print("gt: ", depth_gt[:5, :5])
        occ_bound = occ_bound[80:560, :]

        err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
        errors.append(err)

        err = compute_depth_metrics(depth_pred, depth_gt)
        errors_sharp.append(err)

        err = compute_depth_boundary_error(occ_bound, depth_pred)
        errors_occ.append(err[:2])

    e = np.array(errors).mean(0).tolist()
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_sharp).mean(0).tolist()

    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_occ).mean(0).tolist()
    print("{:>10}, {:>10}".format('dbe_acc', 'dbe_com'))
    print("{:10.4f}, {:10.4f}".format(e[0], e[1]))

def evaluate_solodepth_nyu():

    print("Model evaluation on NYU occlusion boundary dataset!")

    dataset_path = '../NYU_occlusion_boundaries/'
    indices = np.load(dataset_path + 'indices.npy')
    print("Size of NYU occlusion dataset: ", len(indices))

    config = nyu.NYUConfig()
    config.DEPTH_THRESHOLD = 0
    config.PREDICT_DEPTH = True
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.CHAM_LOSS = False
    config.GRAD_LOSS = False
    config.CHAM_COMBINE = False

    depth_model = DepthCNN(config)
    depth_model.cuda()

    #checkpoint_dir = 'checkpoints/scannet20191209T1450/mask_rcnn_scannet_0020.pth'
    checkpoint_dir = 'checkpoints/nyudepth20191207T1213/mask_rcnn_nyudepth_0200.pth'
    depth_model.load_state_dict(torch.load(checkpoint_dir))

    errors = []
    errors_sharp = []
    errors_occ = []

    for i in indices:

        depth = np.load(dataset_path + str(i) + '_depth.npy')/4
        rgb = imread(dataset_path + str(i) + '_img.png')
        occ_bound = imread(dataset_path + str(i) + '_ob.png')

        rgb, window, scale, padding = utils_original.resize_image(
            np.array(rgb),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        depth, _, _, _ = utils_original.resize_depth(
            np.array(depth),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)

        occ_bound, _, _, _ = utils_original.resize_depth(
            np.array(occ_bound),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)


        rgb = mold_image(rgb, config)
        rgb = rgb.transpose(2,0,1)


        gt_depth = torch.tensor(depth).unsqueeze(0)
        rgb = torch.tensor(rgb).unsqueeze(0)

        images = Variable(rgb)
        gt_depth = Variable(gt_depth)

        images = images.cuda().float()
        gt_depth = gt_depth.cuda().float()
        #print("image input shape: ", images.shape)

        depth_np = depth_model.predict([images, gt_depth], mode='inference')

        depth_pred = depth_np[0]
        #print('pred_depth_shape: ', depth_pred.shape)
        depth_pred = depth_pred[0, 80:560, :].detach().cpu().numpy()
        depth_gt = gt_depth[0, 80:560, :].cpu().numpy()
        occ_bound = occ_bound[80:560, :]

        err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
        errors.append(err)

        err = compute_depth_metrics(depth_pred, depth_gt)
        errors_sharp.append(err)

        err = compute_depth_boundary_error(occ_bound, depth_pred, low_thresh=0.03, high_thresh=0.05)
        errors_occ.append(err[:2])

    e = np.array(errors).mean(0).tolist()
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_sharp).mean(0).tolist()

    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_occ).mean(0).tolist()
    print("{:>10}, {:>10}".format('dbe_acc', 'dbe_com'))
    print("{:10.4f}, {:10.4f}".format(e[0], e[1]))

def evaluate_roidepth_scannet():
    print("Model evaluation on NYU occlusion boundary dataset!")

    dataset_path = '../NYU_occlusion_boundaries/'
    indices = np.load(dataset_path + 'indices.npy')
    print("Size of NYU occlusion dataset: ", len(indices))

    '''
    config = scannet.ScannetConfig()

    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.DEPTH_THRESHOLD = 0
    config.GEOMETRIC_LOSS = False
    config.GRAD_LOSS = False
    config.USE_MINI_MASK = False
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28
    '''

    config = nyu.NYUConfig()

    config.DEPTH_THRESHOLD = 0
    config.CHAM_LOSS = False
    config.GRAD_LOSS = False
    config.CHAM_COMBINE = False
    config.PREDICT_DEPTH = True
    config.PREDICT_GLOBAL_DEPTH = True
    config.GEOMETRIC_LOSS = False
    depth_weight = 1
    config.USE_MINI_MASK = True
    config.PREDICT_PLANE = False
    config.PREDICT_NORMAL = False
    config.DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    config.BATCH_SIZE = 1
    config.MASK_SHAPE = [56, 56]
    config.MASK_POOL_SIZE = 28

    model_maskdepth = MaskDepthRCNN(config)
    model_maskdepth.cuda()

    checkpoint_dir = 'checkpoints/scannet20191218T1651/mask_depth_rcnn_scannet_0020.pth'
    model_maskdepth.load_state_dict(torch.load(checkpoint_dir)['model_state_dict'])

    errors = []
    errors_sharp = []
    errors_occ = []

    for i in indices:
        depth = np.load(dataset_path + str(i) + '_depth.npy')
        rgb = imread(dataset_path + str(i) + '_img.png')
        occ_bound = imread(dataset_path + str(i) + '_ob.png')

        rgb, window, scale, padding = utils_original.resize_image(
            np.array(rgb),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)
        depth, _, _, _ = utils_original.resize_depth(
            np.array(depth),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)

        occ_bound, _, _, _ = utils_original.resize_depth(
            np.array(occ_bound),
            min_dim=config.IMAGE_MAX_DIM,
            max_dim=config.IMAGE_MAX_DIM,
            padding=config.IMAGE_PADDING)

        rgb = mold_image(rgb, config)
        rgb = rgb.transpose(2, 0, 1)

        image_meta = compose_image_meta(
            0, rgb.shape, window,
            np.zeros([config.NUM_CLASSES], dtype=np.int32))

        gt_depth = torch.tensor(depth).unsqueeze(0)
        rgb = torch.tensor(rgb).unsqueeze(0)
        image_metas = torch.tensor(image_meta).unsqueeze(0)

        images = Variable(rgb)
        gt_depth = Variable(gt_depth)
        image_metas = Variable(image_metas)

        images = images.cuda().float()
        gt_depth = gt_depth.cuda().float()
        image_metas = image_metas.cuda()
        # print("image input shape: ", images.shape)

        detections, mrcnn_mask, mrcnn_depth, mrcnn_normals, depth_pred = model_maskdepth.predict3([images, image_metas], mode='inference')

        depth_pred = depth_pred[0, 80:560, :].detach().cpu().numpy()
        depth_gt = gt_depth[0, 80:560, :].cpu().numpy()
        occ_bound = occ_bound[80:560, :]

        #print("pred: ", depth_pred[:5, :5])
        #print("gt: ", depth_gt[:5, :5])

        err = evaluateDepths(depth_pred, depth_gt, printInfo=False)
        errors.append(err)

        err = compute_depth_metrics(depth_pred, depth_gt)
        errors_sharp.append(err)

        err = compute_depth_boundary_error(occ_bound, depth_pred)
        errors_occ.append(err[:2])

    e = np.array(errors).mean(0).tolist()
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('rel', 'rel_sqr', 'log_10', 'rmse',
                                                                                  'rmse_log', 'a1', 'a2', 'a3'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_sharp).mean(0).tolist()

    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3],
                                                                                                e[4], e[5], e[6], e[7]))

    e = np.array(errors_occ).mean(0).tolist()
    print("{:>10}, {:>10}".format('dbe_acc', 'dbe_com'))
    print("{:10.4f}, {:10.4f}".format(e[0], e[1]))



if __name__ == '__main__':
    #evaluate_roidepth_scannet()
    evaluate_solodepth_nyu()