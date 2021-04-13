import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
#from keras.models import model_from_json

model = load_model('ASL_model.h5')
pred_list = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']


for j in range(0,26):
    img = cv2.imread('Test_Images/'+pred_list[j]+'_test.jpg')
    #print(img)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
    imgs = imgs.reshape(-1,28,28,3)
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')/255.0
    pred = model.predict_classes(imgs)
    print(pred_list[pred[0]])
