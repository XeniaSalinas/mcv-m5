### Utilities for CAM models.
import numpy as np
from skimage.measure import label


# I am assuming the images come one by one. No batches by now.


def image2bboxes(model, image):
    # Preprocess the image:
    x = preprocess_input(image)
    # Get the heatmaps for all the classes:
    heatmaps = model.predict(x)
    # Get the top 5 scoring classes:
    top_classes = get_top_5_classes(heatmaps)
    # Initialize the bounding boxes:
    bboxes = np.zeros([5,4]) # Five bounding boxes, described by 4 numbers each.
    # Loop over the top 5 classes:
    idx_bbox = -1
    for idx_class in top_classes:
        idx_bbox += 1
        # Get the bounding box for the current class:
        bboxes[idx_bbox,:] = heatmap2bbox(heatmaps[0,:,:,idx_class])
    return bboxes
        
    
def get_top_5_classes(heatmaps):
    # Number of classes:
    nclasses = heatmaps.shape[3]
    # Initialize vector with score of every class:
    scores = np.zeros(nclasses)
    # Loop over classes:
    for idx_class in range(nclasses):
        scores[idx_class] = sum(sum(heatmaps[0,:,:,idx_class]))
    # Fin the top 5 scores:
    scores_aux = scores
    top_classes = np.zeros(5, dtype=np.int16)
    for i in range(5):
        top_classes[i] = np.argmax(scores_aux)
        scores_aux = scores_aux[np.arange(len(scores_aux)) != top_classes[i]]
    return top_classes
    

def heatmap2bbox(heatmap, threshold):
    # Here the heatmap is an array of only two dimensions (the spatial dimensions)
    mask = heatmap > threshold
    # Connected components labeling:
    regions, nregions = label(mask, 8, return_num=True)
    # Keep only the biggest region:
    biggest_region = -1
    biggest_area = 0
    for i in range(nregions):
        current_area = sum(sum(regions == i))
        if current_area > biggest_area:
            biggest_area = current_area
            biggest_region = i
    # Arrays with coordinates of biggest region::
    x_idxs = np.zeros(biggest_area)
    y_idxs = np.zeros(biggest_area)
    count = -1
    for i in regions.shape[0]:
        for j in regions.shape[1]:
            if regions[i,j] == biggest_region:
                count += 1
                x_idxs[count] = j
                y_idxs[count] = i
    # Corners of the bounding box enclosing the biggest region:
    # top left corner:
    x = np.min(x_idxs)
    y = np.min(y_idxs)
    # height and width:
    w = np.max(x_idxs) - x + 1
    h = np.max(y_idxs) - y + 1
    # Join in one array:
    bbox = np.array([x, y, w, h])
    return bbox

