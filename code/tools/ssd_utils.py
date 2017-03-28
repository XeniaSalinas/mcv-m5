import numpy as np
import warnings
import math
import pickle
import tensorflow as tf

"""
    SSD utitlities
"""

overlap_threshold=0.5  # overlap_threshold: Threshold to assign box to a prior.
nms_thresh=0.45        # nms_thresh: Nms threshold.
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
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_width * variances[:, 1]
    decode_bbox_center_y += prior_center_y
    decode_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
    decode_bbox_height *= prior_height
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                  decode_bbox_ymin[:, None],
                                  decode_bbox_xmax[:, None],
                                  decode_bbox_ymax[:, None]), axis=-1)
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox


def detection_out(predictions, num_classes, background_label_id=0, keep_top_k=200,
                  confidence_threshold=0.01):
    """Do non maximum suppression (nms) on prediction results.
    # Arguments
        predictions: Numpy array of predicted values.
        num_classes: Number of classes for prediction.
        background_label_id: Label of background class.
        keep_top_k: Number of total bboxes to be kept per image
            after nms step.
        confidence_threshold: Only consider detections,
            whose confidences are larger than a threshold.
    # Return
        results: List of predictions for every picture. Each prediction is:
            [label, confidence, xmin, ymin, xmax, ymax]
    """
    boxes = tf.placeholder(dtype='float32', shape=(None, 4))
    scores = tf.placeholder(dtype='float32', shape=(None,))
    nms = tf.image.non_max_suppression(boxes, scores,
                                            top_k,
                                            iou_threshold=nms_thresh)
    sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    mbox_loc = predictions[:, :, :4]
    variances = predictions[:, :, -4:]
    mbox_priorbox = predictions[:, :, -8:-4]
    mbox_conf = predictions[:, :, 4:-8]
    results = []
    for i in range(len(mbox_loc)):
        results.append([])
        decode_bbox = decode_boxes(mbox_loc[i],
                                        mbox_priorbox[i], variances[i])
        for c in range(num_classes):
            if c == background_label_id:
                continue
            c_confs = mbox_conf[i, :, c]
            c_confs_m = c_confs > confidence_threshold
            if len(c_confs[c_confs_m]) > 0:
                boxes_to_process = decode_bbox[c_confs_m]
                confs_to_process = c_confs[c_confs_m]
                feed_dict = {boxes: boxes_to_process,
                             scores: confs_to_process}
                idx = sess.run(nms, feed_dict=feed_dict)
                good_boxes = boxes_to_process[idx]
                confs = confs_to_process[idx][:, None]
                labels = c * np.ones((len(idx), 1))
                c_pred = np.concatenate((labels, confs, good_boxes),
                                        axis=1)
                results[-1].extend(c_pred)
        if len(results[-1]) > 0:
            results[-1] = np.array(results[-1])
            argsort = np.argsort(results[-1][:, 1])[::-1]
            results[-1] = results[-1][argsort]
            results[-1] = results[-1][:keep_top_k]
    return results
