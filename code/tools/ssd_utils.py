import numpy as np
import warnings
import math
import cv2

"""
    SSD utitlities
"""

def ssd_build_gt_batch(batch_gt,image_shape,num_classes):

    batch_size = len(batch_gt)
    num_boxes = 7308
    batch_y = np.zeros([batch_size,num_boxes,4+num_classes+8])

    for i,gt in enumerate(batch_gt):
        if gt.shape[0] == 0:
          # if there are no objects we'll get NaNs on SSDLoss, set everything to one!
          # TODO check if the following line harms learning in case of 
          #      having lots of images with no objects
          batch_y[i] = np.ones((num_boxes,4+num_classes+8))
          continue
        objects = gt.tolist()
        for j,obj in enumerate(objects):
            centerx = obj[1] * image_shape[2]
            centery = obj[2] * image_shape[1]
            width = obj[3] * image_shape[2]
            heigth = obj[4] * image_shape[1]
            box = np.zeros(4 + num_classes + 8)
            box[0] = centerx
            box[1] = centery
            box[2] = width
            box[3] = heigth
            box[int(obj[0])+4] = 1
            batch_y[i,j,:] = box

    return batch_y
