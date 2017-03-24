### Utilities for CAM models.
import numpy as np
from skimage.measure import label


# I am assuming the images come one by one. No batches by now.


def image2bboxes(model, image, percent):
    # Preprocess the image:
    x = preprocess_input(image)
    # Get the heatmaps for all the classes:
    heatmaps = model.predict(x)
    # Get the top 5 scoring classes:
    top_classes = get_top_5_classes(heatmaps)
    
    print 'top_classes = ' + str(top_classes)
    
    # Initialize the bounding boxes:
    bboxes = np.zeros([5,4]) # Five bounding boxes, described by 4 numbers each.
    # Loop over the top 5 classes:
    idx_bbox = -1
    for idx_class in top_classes:
        idx_bbox += 1
        # Get the bounding box for the current class:
        bboxes[idx_bbox,:] = heatmap2bbox(heatmaps[0,:,:,idx_class], percent)
        print 'bbox = ' + str(bboxes[idx_bbox,:])
    return bboxes
        
    
def get_top_5_classes(heatmaps):
    # Number of classes:
    nclasses = heatmaps.shape[3]
    # Initialize vector with score of every class:
    scores = np.zeros(nclasses)
    # Loop over classes:
    for idx_class in range(nclasses):
        scores[idx_class] = sum(sum(heatmaps[0,:,:,idx_class]))
        
    print 'scores = ' + str(scores)
        
    # Fin the top 5 scores:
    scores_aux = scores
    top_classes = np.zeros(5, dtype=np.int16)
    for i in range(5):
        top_classes[i] = np.argmax(scores_aux)
        scores_aux = scores_aux[np.arange(len(scores_aux)) != top_classes[i]]
    return top_classes
    

def heatmap2bbox(heatmap, percent):
    # Compute the threshold from the percentage and the heatmap:
    threshold = np.percentile(heatmap.flatten(), percent)
    print 'min(heatmap) = ' + str(np.min(heatmap.flatten()))
    print 'max(heatmap) = ' + str(np.max(heatmap.flatten()))
    print 'threshold = ' + str(threshold)
    # Here the heatmap is an array of only two dimensions (the spatial dimensions)
    mask = heatmap > threshold
    print mask
    # Connected components labeling:
    regions, nregions = label(mask, 8, return_num=True)
    print 'regions.__class__.__name__ = ' + regions.__class__.__name__
    print 'regions.shape = ' + str(regions.shape)
    print 'nregions = ' + str(nregions)
    print regions
    if nregions > 0:
        # Keep only the biggest region:
        biggest_region = -1
        biggest_area = 0
        # We are discarding region 0, wich corresponds to background
        for i in range(1,nregions+1):
            print i
            current_area = sum(sum(regions == i))
            if current_area > biggest_area:
                biggest_area = current_area
                biggest_region = i
        print 'biggest area = ' + str(biggest_area)
        # Arrays with coordinates of biggest region::
        x_idxs = np.zeros(biggest_area)
        y_idxs = np.zeros(biggest_area)
        count = -1
        for i in range(regions.shape[0]):
            for j in range(regions.shape[1]):
                if regions[i,j] == biggest_region:
                    count += 1
                    x_idxs[count] = j
                    y_idxs[count] = i
        # Corners of the bounding box enclosing the biggest region:
        # top left corner:
        print 'x_idxs = ' + str(x_idxs)
        print 'y_idxs = ' + str(y_idxs)
        x = np.min(x_idxs)
        y = np.min(y_idxs)
        # height and width:
        w = np.max(x_idxs) - x + 1
        h = np.max(y_idxs) - y + 1
        # Join in one array:
        bbox = np.array([x, y, w, h])
        
    else:
        bbox = np.array([-1, -1, -1, -1])
    
    return bbox

def preprocess_input(x):
    return x
