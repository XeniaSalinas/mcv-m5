# predict with CAM model
from models.vggGAP_pred import build_vggGAP_pred
from tools.cam_utils import image2bboxes
from keras.preprocessing import image



#model = build_vggGAP_pred(img_shape=(3, 224, 224), n_classes=45)
model = build_vggGAP_pred(img_shape=(224, 224, 3), n_classes=221)


model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['recall'])


img_path = '/home/xianlopez/Documents/myvenv1/23_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))

bboxes = image2bboxes(model, img)




