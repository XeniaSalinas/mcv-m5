# predict with CAM model
from models.vggGAP_pred import build_vggGAP_pred
from tools.cam_utils import image2bboxes
from keras.preprocessing import image



model = build_vggGAP_pred(img_shape=(3, 224, 224), n_classes=45)


model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['recall'])


img_path = './../data/Datasets/detection/TT100K_detection/train/23_0.jpg'
img = image.load_img(img_path, target_size=(224, 224))

bboxes = image2bboxes(model, img)




