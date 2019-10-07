"""
Mask R-CNN
Train on the SUN dataset.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Mask RCNN is Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sun.py train --dataset=/Users/ekaterina/Desktop/diploma/mask_rcnn/datasets/SUNRGBD/train --weights=coco


    # Resume training a model that you had trained earlier
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=last

    # Train a new model starting from ImageNet weights
    python3 sun.py train --dataset=/path/to/sun/dataset/train --weights=imagenet

    # Apply color splash to an image
    python3 sun.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 sun.py splash --weights=last --video=<URL or path to file>ß
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2

from config import Config
import utils


############################################################
#  Configurations
############################################################


class SunConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sun"

    # Depth Prediction options
    PREDICT_DEPTH = True
    DEPTH_LOSS = 'L1'  # Options: L1, L2, BERHU
    DEPTH_THRESHOLD = 0.1
    USE_ROI_MINI_MASK = False

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 37  # Background + balloon

    TRAIN_ROIS_PER_IMAGE = 200

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BILINEAR_UPSAMPLING = True


############################################################
#  Dataset
############################################################

class SunDataset(utils.Dataset):

    def load_sun(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.

        class_names = ['wall', 'floor', 'cabinet', 'bed', 'chair','sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floo mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'showe curtain', 'box', 'whiteboard', 'person', 'nigh stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

        if len(class_names) == 0:
            self.add_class("sun", 1, "bed")
            self.add_class("sun", 2, "books")
            self.add_class("sun", 3, "ceiling")
            self.add_class("sun", 4, "chair")
            self.add_class("sun", 5, "floor")
            self.add_class("sun", 6, "furniture")
            self.add_class("sun", 7, "objects")
            self.add_class("sun", 8, "picture")
            self.add_class("sun", 9, "sofa")
            self.add_class("sun", 10, "table")
            self.add_class("sun", 11, "tv")
            self.add_class("sun", 12, "wall")
            self.add_class("sun", 13, "window")
        else:
            n = len(class_names)
            for i in range(n):
                self.add_class("sun", i+1, class_names[i])

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            objects = [s['region_attributes'] for s in a['regions'].values()]
            class_ids = [int(n['class']) for n in objects]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            depth_file_name = a['filename'].rsplit( ".", 1 )[ 0 ] +"_depth.png"
            depth_path = os.path.join(dataset_dir, depth_file_name)
            #depth = skimage.io.imread(depth_path)

            normal_file_name= a['filename'].rsplit( ".", 1 )[ 0 ] +"_plane.npy"
            normal_path = os.path.join(dataset_dir, normal_file_name)


            self.add_image(
                "sun",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                depth_path=depth_path,
                normal_path=normal_path,
                width=width, height=height,
                polygons=polygons,
                class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "sun":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = info['class_ids']
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        masks = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        #print("prepared mask shape: ", masks.shape)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            #print("r c: ",rr, cc)
            #mask[rr, cc, i] = 1
            y_s = p['all_points_y']
            x_s = p['all_points_x']
            points = np.array([x_s, y_s], 'int32').transpose()
            mask = np.zeros([info["height"], info["width"]])
            cv2.fillPoly(mask, [points], 1)
            masks[:, :, i] = mask

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids, dtype=np.int32)
        return masks.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "sun":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


