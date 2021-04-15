import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time
from keras.models import model_from_json
import numpy as np
from tensorflow import keras
from keras.models import load_model



#Set up GUI
window = tk.Tk()  #Makes main window
#window = Tk()
#window.title("Reading Sign Language")
window.wm_title("Reading Hand Signals")
window.config(background="#FFFFFF")
#canvas = Canvas(height=1000, width=1000, bg='#B1DDC6', highlightbackground="#B1DDC6")

#Title in window


#Graphics window
#imageFrame = canvas.create_image(100,100)
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=1, column=0, padx=10, pady=2)


#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=1, column=0)

#start video feed code
video = cv2.VideoCapture(0)

upper_left = (350, 50)
bottom_right = (600, 350)

model = keras.models.load_model('D:/Grad/CSCE 5214 Soft dev for AI/ASL_recognition-1/ASL_dataset_Model weights/ASL_model.h5')
pred_list = ['A','B','C','D','del','E','F','G','H','I','J','K','L','M','N','nothing','O','P','Q','R','S','space','T','U','V','W','X','Y','Z']

def gray_live_video():
    static_back = None
    while True:
        check , frame = video.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ROI = frame[50:350,350:600]
        ROI = cv2.flip(ROI,1)
        r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 1)
        gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if static_back is None: 
            static_back = gray 
            continue

        diff_ROI = cv2.absdiff(static_back, gray) 

        # thresh_ROI = cv2.threshold(diff_ROI, 15, 255, cv2.THRESH_BINARY)[1] 
        # thresh_ROI = cv2.dilate(thresh_ROI, None, iterations = 2) 

        thresh_ROI= cv2.threshold(diff_ROI, 45, 255, cv2.THRESH_BINARY)[1]
        thresh_ROI= cv2.erode(thresh_ROI, None, iterations=2)
        thresh_ROI= cv2.dilate(thresh_ROI, None, iterations=2)

        cnts,_ = cv2.findContours(thresh_ROI.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.imshow("fa",thresh_ROI)
        
        for contour in cnts: 
            ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
            cv2.imshow("output",ROI)
            resized = cv2.resize(ROI, (28,28), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("Output.jpg",ROI)

            # predict the model
            imgs = resized.reshape(-1,28,28,3)
            imgs = np.array(imgs)
            imgs = imgs.astype('float32')/255.0
            pred = model.predict_classes(imgs)
            print(pred_list[pred[0]])

            tk.Label(window,text=pred_list[pred[0]],font=(None, 30)).grid(row=3,column=0)
          
        return frame


def show_frame():
    img = Image.fromarray(gray_live_video())
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)  


tk.Label(window,text="Sign Language Recognition",font=(None, 40)).grid(row=0,column=0)
tk.Label(window,text="English Alphabet",font=(None, 30)).grid(row=2,column=0)
tk.Label(window,text="Model trained using MNIST dataset",font=(None, 10)).grid(row=4,column=0)

show_frame() 
#canvas.pack()

window.mainloop()  #Starts GUI