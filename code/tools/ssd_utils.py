import numpy as np
import warnings
import math
import pickle
import tensorflow as tf

from detection_utils import *

"""
    SSD utitlities
"""

overlap_threshold=0.5  # overlap_threshold: Threshold to assign box to a prior.
top_k=400              # top_k: Number of total bboxes to be kept per image after nms step.     
                 
def IoU(box, priors):
    """Compute intersection over union for the box with all priors.
    # Arguments
        box: Box, numpy tensor of shape (4,).
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
    # Return
        iou: Intersection over union,
            numpy tensor of shape (num_priors).
    """
    # compute intersection
    inter_upleft = np.maximum(priors[:, :2], box[:2])
    inter_botright = np.minimum(priors[:, 2:4], box[2:])
    inter_wh = inter_botright - inter_upleft
    inter_wh = np.maximum(inter_wh, 0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    # compute union
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (priors[:, 2] - priors[:, 0])
    area_gt *= (priors[:, 3] - priors[:, 1])
    union = area_pred + area_gt - inter
    # compute iou
    iou = inter / union
    return iou
        
def encode_box(box, priors, return_iou=True):
    """Encode box for training, do it only for assigned priors.
    # Arguments
        box: Box, numpy tensor of shape (4,).
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
        return_iou: Whether to concat iou to encoded values.
    # Return
        encoded_box: Tensor with encoded box
            numpy tensor of shape (num_priors, 4 + int(return_iou)).
    """
    iou = IoU(box, priors)
    num_priors = 0 if priors is None else len(priors)
    encoded_box = np.zeros((num_priors, 4 + return_iou))
    assign_mask = iou > overlap_threshold
    if not assign_mask.any():
        assign_mask[iou.argmax()] = True
    if return_iou:
        encoded_box[:, -1][assign_mask] = iou[assign_mask]
    assigned_priors = priors[assign_mask]
    box_center = 0.5 * (box[:2] + box[2:])
    box_wh = box[2:] - box[:2]
    assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
    assigned_priors_wh = (assigned_priors[:, 2:4] - assigned_priors[:, :2])
    # we encode variance
    encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
    encoded_box[:, :2][assign_mask] /= assigned_priors_wh
    encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
    encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)
    encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]
    return encoded_box.ravel()
        
def assign_boxes(boxes, num_classes, priors):
    """Assign boxes to priors for training.
    # Arguments
        boxes: Box, numpy tensor of shape (num_boxes, 4 + num_classes),
            num_classes without background.
        num_classes: Number of classes including background.
        priors: Priors and variances, numpy tensor of shape (num_priors, 8),
            priors[i] = [xmin, ymin, xmax, ymax, varxc, varyc, varw, varh].
    # Return
        assignment: Tensor with assigned boxes,
            numpy tensor of shape (num_boxes, 4 + num_classes + 8),
            priors in ground truth are fictitious,
            assignment[:, -8] has 1 if prior should be penalized
                or in other words is assigned to some ground truth box,
            assignment[:, -7:] are all 0. See loss for more details.
    """
    num_priors = 0 if priors is None else len(priors)
    assignment = np.zeros((num_priors, 4 + num_classes + 8))
    assignment[:, 4] = 1.0
    if len(boxes) == 0:
        return assignment

    encoded_boxes = np.apply_along_axis(encode_box, 1, boxes[:, :4], priors)
    encoded_boxes = encoded_boxes.reshape(-1, num_priors, 5)
    best_iou = encoded_boxes[:, :, -1].max(axis=0)
    best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
    best_iou_mask = best_iou > 0
    best_iou_idx = best_iou_idx[best_iou_mask]
    assign_num = len(best_iou_idx)
    encoded_boxes = encoded_boxes[:, best_iou_mask, :]
    assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:4]
    assignment[:, 4][best_iou_mask] = 0
    assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
    assignment[:, -8][best_iou_mask] = 1
    return assignment
    

def ssd_build_gt_batch(batch_gt,image_shape,num_classes):
    """Generate ground truth batch for training
    # Arguments
        batch_gt: griund truth batch
        image_shape: Shape of the input image
        num_classes: Number of classes excluding background.

    # Return
        batch_y: Tensor with assigned boxes,
            numpy tensor of shape (batch_size,num_boxes, 4 + num_classes + 8),
            priors in ground truth are fictitious,
            assignment[:, -8] has 1 if prior should be penalized
                or in other words is assigned to some ground truth box,
            assignment[:, -7:] are all 0. See loss for more details.
    """
    
    priors = pickle.load(open('tools/prior_boxes_ssd300.pkl', 'rb'))
    
    num_boxes=7308
    
    batch_size = len(batch_gt)
    batch_y = np.zeros([batch_size,num_boxes,4 + num_classes + 1 + 8])
    for i,gt in enumerate(batch_gt):
        objects = gt.tolist()
        boxes = np.zeros((len(objects),4 + num_classes))
        for j,obj in enumerate(objects):
            box = np.zeros(4 + num_classes)
            box[0] = obj[1] - (obj[3]/2) # xmin
            box[1] = obj[2] - (obj[4]/2) # ymin
            box[2] = obj[1] + (obj[3]/2) # xmax
            box[3] = obj[2] + (obj[4]/2) # ymax
            box[int(obj[0]) + 4] = 1 # class to one hot
            boxes[j]=box 
        assigned = assign_boxes(boxes, num_classes+1, priors)  # num_classes+1 to include background  
        batch_y[i,:] = assigned
   
    return batch_y

def decode_boxes(mbox_loc, mbox_priorbox, variances):
    """Convert bboxes from local predictions to shifted priors.
    # Arguments
        mbox_loc: Numpy array of predicted locations.
        mbox_priorbox: Numpy array of prior boxes.
        variances: Numpy array of variances.
    # Return
        decode_bbox: Shifted priors.
    """
    prior_width = mbox_priorbox[2] - mbox_priorbox[0]
    prior_height = mbox_priorbox[3] - mbox_priorbox[1]
    prior_center_x = 0.5 * (mbox_priorbox[2] + mbox_priorbox[0])
    prior_center_y = 0.5 * (mbox_priorbox[3] + mbox_priorbox[1])
    decode_bbox_center_x = mbox_loc[0] * prior_width * variances[0]
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[1] * prior_width * variances[1]
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(mbox_loc[2] * variances[2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[3] * variances[3])
    decode_bbox_height *= prior_height
    decode_bbox = np.concatenate((decode_bbox_center_x[None],
                                  decode_bbox_center_y[None],
                                  decode_bbox_width[None],
                                  decode_bbox_height[None]), axis=-1)
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox


def ssd_detection_out(predictions, labels, threshold, nms_threshold, background_label_id=0):
    """Do non maximum suppression (nms) on prediction results.
    # Arguments
        predictions: Numpy array of predicted values.
        labels: list of labels
        threshold: Min probablity for a prediction to be considered
        nms_threshold: Non Maximum Suppression threshold
        background_label_id: Label of background class.
    # Return
        boxes: List of predictions for every picture. Each prediction is a BoundBox 
    """
    C = len(labels)
    boxes = list()
    
    mbox_loc = predictions[:, :4]
    variances = predictions[:, -4:]
    mbox_priorbox = predictions[:, -8:-4]
    mbox_conf = predictions[:, 4:-8]
    
    for i in range(len(mbox_loc)):
        bx = BoundBox(C)
        decode_bbox = decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])
        bx.x = decode_bbox[0]
        bx.y = decode_bbox[1]
        bx.w = decode_bbox[2]
        bx.h = decode_bbox[3]
        bx.probs = mbox_conf[i,1:]
        bx.probs *= bx.probs > threshold
        boxes.append(bx)
    
    # non max suppress boxes
    for c in range(C):
        for i in range(len(boxes)):
            boxes[i].class_num = c
        boxes = sorted(boxes, key = prob_compare)
        for i in range(len(boxes)):
            boxi = boxes[i]
            if boxi.probs[c] == 0: continue
            for j in range(i + 1, len(boxes)):
                boxj = boxes[j]
                if box_iou(boxi, boxj) >= nms_threshold:
                    boxes[j].probs[c] = 0.
    return boxes