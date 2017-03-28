### Utilities for CAM models.
import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt


# I am assuming the images come one by one. No batches by now.


def image2bboxes(model, image, percent, K, rescale):
    # Preprocess the image:
    x = preprocess_input(image, rescale)
    # Get the heatmaps for all the classes:
    heatmaps = model.predict(x)
    # Get the top K scoring classes:
    top_classes = get_top_K_classes(heatmaps, K)
    
    # Write the heatmap (if we are only considering one top class):
    if K == 1:
        write_heatmap(heatmaps[0,:,:,top_classes[0]])
    
    # Initialize the bounding boxes:
    bboxes = np.zeros([K,4]) # K bounding boxes, described by 4 numbers each.
    # Loop over the top 5 classes:
    idx_bbox = -1
    for idx_class in top_classes:
        idx_bbox += 1
        # Get the bounding box for the current class:
        bboxes[idx_bbox,:] = heatmap2bbox(heatmaps[0,:,:,idx_class], percent)
    return bboxes
        
    
# Find the K classes that have largest prediction score.
def get_top_K_classes(heatmaps, K):
    # Number of classes:
    nclasses = heatmaps.shape[3]
    # Initialize vector with score of every class:
    scores = np.zeros(nclasses)
    # Loop over classes:
    for idx_class in range(nclasses):
        scores[idx_class] = sum(sum(heatmaps[0,:,:,idx_class]))
        
    # Fin the top K scores:
    scores_aux = scores
    top_classes = np.zeros(K, dtype=np.int16)
    for i in range(K):
        top_classes[i] = np.argmax(scores_aux)
        scores_aux = scores_aux[np.arange(len(scores_aux)) != top_classes[i]]
    return top_classes
    

def heatmap2bbox(heatmap, percent):
    # Compute the threshold from the percentage and the heatmap:
    # TODO: in the paper the use as threshold the 20% of the maximum. Here I 
    # cannot do this, since we have negative values...
    threshold = np.percentile(heatmap.flatten(), percent)
    # Here the heatmap is an array of only two dimensions (the spatial dimensions)
    mask = heatmap > threshold
    # Write mask:
    write_mask(mask)
    # Connected components labeling:
    regions, nregions = label(mask, 8, return_num=True)
    if nregions > 0:
        # Keep only the biggest region:
        biggest_region = -1
        biggest_area = 0
        # We are discarding region 0, wich corresponds to background
        for i in range(1,nregions+1):
            current_area = sum(sum(regions == i))
            if current_area > biggest_area:
                biggest_area = current_area
                biggest_region = i
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

def preprocess_input(x, rescale):
    # 'RGB'->'BGR'
    x = x[:, :, ::-1]
    # Zero-center by mean pixel
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    # Rescale>
    x *= rescale
    return x


def plot_image_n_bboxes(img, bboxes):
    plt.matshow(img)
    plt.show()


# Write tatrix to text file:
def write_heatmap(array_in):
    matrix = np.matrix(array_in)
    with open('heatmap.txt', 'w') as f:
        for line in matrix:
            np.savetxt(f, line, fmt='%.2f')


# Write tatrix to text file:
def write_mask(array_in):
    matrix = np.matrix(array_in)
    with open('mask.txt', 'w') as f:
        for line in matrix:
            np.savetxt(f, line, fmt='%i')
    
    
    
    