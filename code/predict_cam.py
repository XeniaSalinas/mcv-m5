# predict with CAM model
from models.vggGAP_pred import build_vggGAP_pred
from tools.cam_utils import image2bboxes
from keras.preprocessing import image
import numpy as np



#model = build_vggGAP_pred(img_shape=(3, 224, 224), n_classes=45)
model = build_vggGAP_pred(img_shape=(224, 224, 3), n_classes=221)


model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['recall'])

# Percentile to establish the threshold:
percent = 90

#img_path = '/home/xianlopez/Documents/myvenv1/23_0.jpg'
img_path = '/home/xianlopez/Documents/myvenv1/45598_10.jpg'
#img_path = '/home/xianlopez/Documents/myvenv1/45624_1.jpg'

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

print 'x.__class__.__name__ = ' + x.__class__.__name__
print 'x.shape= ' + str(x.shape)
bboxes = image2bboxes(model, x, percent)




