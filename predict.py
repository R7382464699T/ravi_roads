import glob

import cv2
import numpy as np
from keras.models import load_model
from roads import dice_coef,dice_coef_loss

WIDTH = 224
HEIGHT = 224

model = load_model(
    filepath='model_weights_roadssss.hdf5',
    custom_objects={'dice_coef': dice_coef,'dice_coef_loss': dice_coef_loss}
)

for path in glob.iglob('/home/asus/Documents/satellite/mass_roads/test/sat/*.tiff',
                       recursive=True):
    try:
        # path = "/home/mv/Desktop/documents/E-state_png/514251507Jun20161.png"
        filename = path.split("/")[-1]
        file = filename.split(".jpg")[0]

        #########for overlay and to know the shape of the orginal image#######
        test_img1 = cv2.imread(path, 0)
        ###########################

        test_img = cv2.resize(cv2.imread(path), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        # cv2.imwrite("/home/mv/Desktop/documents/Test (2)/icdr_table_outputs/" + str(file) + "_copy" + ".png", test_img1)
        test_img = np.expand_dims(test_img, axis=0)
        test_img = np.array(test_img, np.float32) / 255

        preds = model.predict(test_img)
        preds = np.squeeze(preds)

        shape = test_img1.shape

        width1 = 1500
        height1 = 1500

        prob = cv2.resize(preds, (width1, height1))
        mask = prob > 0.5
        mask1 = np.multiply(mask, 255)
        cv2.imwrite("/home/asus/Documents/satellite/mass_roads/test/mask/" + str(file) + "mask_out_f" + ".png",mask1)
    except Exception as e:
        print(str(e))
