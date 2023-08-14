import tkinter as tk
from tkinter import filedialog
from tkinter import *
import time

from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top = tk.Tk()
top.geometry('1000x800')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model_a1.json","model_weights1.h5")

EMOTIONS_LIST = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_image,1.3,5)
    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
        print("Predicted Emotion is" + pred)
        label1.configure(foreground="#011638",text = pred)
    except:
        label1.configure(foreground="#011638",text = "Unable to detect")


def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Emotion", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass


def realTimeCapture():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4556)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
    frame_interval = 2  # Process every 2 frames
    prev_time = time.time()
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        
        if int((time.time() - prev_time) * 1000) >= frame_interval:
            prev_time = time.time()
            # Preprocess the frame (resize, convert to grayscale, normalize)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_frame,1.3,5)

            for (x,y,w,h) in faces:
                fc = gray_frame[y:y+h,x:x+w]
                roi = cv2.resize(fc,(48,48))
                pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
                cv2.putText(frame, pred, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 7)
            
        
        # Display the predicted emotion on the frame
        # cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame with emotion prediction
        cv2.putText(frame, "Press the Q key on the keyboard to exit", (10, 575), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the window
    cap.release()
    cv2.destroyAllWindows()


#My OWN CODE----------------------------------------------------------------------------
def captureImage():
    cam_port = 0

    cam = cv2.VideoCapture(cam_port)

    
    if (cam.isOpened()== False):
        print("Error opening video file")

    result, image = cam.read()

    # If image will detected without any error,
    # show result
    if result:

        cv2.imshow("CamPicture", image)

        # saving image in local storage
        cv2.imwrite("CamPicture.png", image)

        # If keyboard interrupt occurs, destroy image
        # window
        cv2.waitKey(0)
        # cv.destroyWindow("CamPicture")

        try:
            file_path = "CamPicture.png"
            uploaded = Image.open(file_path)
            uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
            im = ImageTk.PhotoImage(uploaded)

            sign_image.configure(image=im)
            sign_image.image = im
            label1.configure(text='')
            show_Detect_button(file_path)
        except:
            pass

    # If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
#---------------------------------------------------------------------------------------


upload1 = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload1.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload1.pack(side='bottom',pady=50)
upload1.place(x=25, y=100)

#My OWN CODE----------------------------------------------------------------------------
upload2 = Button(top, text="Open Front Video", command=realTimeCapture, padx=10, pady=5)
upload2.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload2.pack(side='bottom',pady=50)
upload2.place(x=25, y=300)

upload3 = Button(top, text="Open Front Camera", command=captureImage, padx=10, pady=5)
upload3.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload3.pack(side='bottom',pady=50)
upload3.place(x=25, y=500)
#---------------------------------------------------------------------------------------

sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Emotion Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()

top.mainloop()
