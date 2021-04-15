import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
<<<<<<< HEAD
from keras.models import model_from_json

model = keras.models.load_model('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/ASL_model.h5')
pred_list = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']


model = keras.models.load_model('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/ASL_Model_Weights.h5')
#model = load_model('ASL_Model_Weights.h5')
pred_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

for j in range(0,2):
    # img = cv2.imread('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/Test_Images/'+pred_list[j]+'_test.jpg')
    # #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgs = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
    # imgs = imgs.reshape(-1,28,28,3)
    
    #imgs = cv2.imread("D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/Output.png")
    imgs = cv2.imread("D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/Test_Images/X_test.jpg")
    imgs = cv2.resize(imgs, (28,28), interpolation = cv2.INTER_CUBIC)
    imgs = imgs.reshape(-1,28,28,3)
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')/255.0

    pred = np.argmax(model.predict_classes(imgs), axis=-1)
    print(pred)
