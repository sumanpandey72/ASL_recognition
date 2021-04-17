import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from keras.models import model_from_json


model = keras.models.load_model('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/Model_training/model_weights_improved.h5')
pred_list = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# for j in range(0,2):
#     # img = cv2.imread('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/Test_Images/'+pred_list[j]+'_test.jpg')
#     # #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # imgs = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
#     # imgs = imgs.reshape(-1,28,28,3)
    
#     #imgs = cv2.imread("D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/Output.png")
#     imgs = cv2.imread("D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/resized_output.jpg")
#     #imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)
#     # imgs = cv2.resize(imgs, (28,28))
#     # cv2.imwrite("Output_ASL.jpg",imgs)
#     imgs = imgs.reshape(-1,28,28,1)
#     imgs = np.array(imgs)
#     imgs = imgs.astype('float32')/255.0

#     pred = model.predict_classes(imgs)
#     print(pred_list[pred[0]])

for j in range(0,17):
    img = cv2.imread('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/hand/'+pred_list[j]+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgs = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
    imgs = imgs.reshape(-1,28,28,1)
    imgs = np.array(imgs)
    imgs = imgs.astype('float32')/255.0
    pred = model.predict_classes(imgs)
    print(pred_list[pred[0]])