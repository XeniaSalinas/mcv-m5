import os,sys
import cv2
import numpy as np
from tools.detection_utils import *

# Input parameters to select the Dataset
dataset_name = 'Udacity' #accepted datasets: Udacity or TT100K_detection

def draw_detections(boxes, im, labels):

        def get_color(c,x,max):
          colors = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) )
          ratio = (float(x)/max)*5
          i = np.floor(ratio)
          j = np.ceil(ratio)
          ratio -= i
          r = (1-ratio) * colors[int(i)][int(c)] + ratio*colors[int(j)][int(c)]
          return r*255

	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	for b in boxes:
		max_indx = np.argmax(b.probs)
		max_prob = b.probs[max_indx]
		label = 'object' * int(len(labels) < 2)
		label += labels[max_indx] * int(len(labels)>1)
		if max_prob > 0.5:
			left  = int ((b.x - b.w/2.) * w)
			right = int ((b.x + b.w/2.) * w)
			top   = int ((b.y - b.h/2.) * h)
			bot   = int ((b.y + b.h/2.) * h)
			if left  < 0    :  left = 0
			if right > w - 1: right = w - 1
			if top   < 0    :   top = 0
			if bot   > h - 1:   bot = h - 1
			thick = int((h+w)/300)
			mess = '{}'.format(label)
                        offset = max_indx*123457 % len(labels)
                        color = (get_color(2,offset,len(labels)),
                                 get_color(1,offset,len(labels)),
                                 get_color(0,offset,len(labels)))
			cv2.rectangle(imgcv,
				(left, top), (right, bot),
				color, thick)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        scale = 0.65
                        thickness = 1
                        size=cv2.getTextSize(mess, font, scale, thickness)
                        cv2.rectangle(im, (left-2,top-size[0][1]-4), (left+size[0][0]+4,top), color, -1)
                        cv2.putText(im, mess, (left+2,top-2), font, scale, (0,0,0), thickness, cv2.LINE_AA)
	return imgcv

if len(sys.argv) < 2:
  print "USAGE: python eval_detection_fscore.py path_to_images"
  quit()

if dataset_name == 'TT100K_detection':
    classes = ['i2','i4','i5','il100','il60','il80','io','ip','p10','p11','p12','p19','p23','p26','p27','p3','p5','p6','pg','ph4','ph4.5','ph5','pl100','pl120','pl20','pl30','pl40','pl5','pl50','pl60','pl70','pl80','pm20','pm30','pm55','pn','pne','po','pr40','w13','w32','w55','w57','w59','wo']
elif dataset_name == 'Udacity':
    classes = ['Car','Pedestrian','Truck']
else:
    print "Error: Dataset not found!"
    quit()

test_dir = sys.argv[1]
imfiles = [os.path.join(test_dir,f) for f in os.listdir(test_dir) 
                                    if os.path.isfile(os.path.join(test_dir,f)) 
                                    and f.endswith('jpg')]

if len(imfiles) == 0:
  print "ERR: path_to_images do not contain any jpg file"
  quit()
  
img_paths = []
chunk_size = 128 # we are going to process all image files in chunks

for i,img_path in enumerate(imfiles):

  img_paths.append(img_path)

  if len(img_paths)%chunk_size == 0 or i+1 == len(imfiles):

    for i,img_path in enumerate(img_paths):

        boxes_true = []
        label_path = img_path.replace('jpg','txt')
        gt = np.loadtxt(label_path)
        if len(gt.shape) == 1:
          gt = gt[np.newaxis,]
        for j in range(gt.shape[0]):
          bx = BoundBox(len(classes))
          bx.probs[int(gt[j,0])] = 1.
          bx.x, bx.y, bx.w, bx.h = gt[j,1:].tolist()
          boxes_true.append(bx)
       
        # You can visualize/save per image results with this:
        if img_path.find('/test') != -1: #only for test
          im = cv2.imread(img_path)
          im = draw_detections(boxes_true, im, classes)
          #cv2.imshow('', im)
          #cv2.waitKey(0)
          img_name = img_path[img_path.rfind('/'):]
          if not os.path.exists(sys.argv[1] + 'ground_truth'):
            os.makedirs(sys.argv[1] + 'ground_truth')
          cv2.imwrite(sys.argv[1] + '/ground_truth' + img_name,im)

    img_paths = []