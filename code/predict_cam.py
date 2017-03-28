# predict with CAM model
from models.vggGAP_pred import build_vggGAP_pred
from tools.cam_utils import image2bboxes
from keras.preprocessing import image
import numpy as np



#model = build_vggGAP_pred(img_shape=(3, 224, 224), n_classes=45)
model = build_vggGAP_pred(img_shape=(224, 224, 3), n_classes=45)


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Percentile to establish the threshold:
percent = 90

# Rescale images factor:
rescale = 1/255.

#img_path = '/home/xianlopez/Documents/myvenv1/23_0.jpg'
#img_path = '/home/xianlopez/Documents/myvenv1/45598_10.jpg'
img_path = '/home/xianlopez/Documents/myvenv1/45624_1.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Number of bounding boxes to generate:
K = 1
# Apply the model and generate the boxes:
bboxes = image2bboxes(model, x, percent, K, rescale)




