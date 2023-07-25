#Importing Necessary Libraries
import tkinter as tk 
from tkinter import filedialog, messagebox
from tkinter import *
from PIL import Image,ImageTk
import numpy
import numpy as np
from collections import Counter
from scipy.stats import mode
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

#Loading the Model
from keras.models import load_model
model = load_model('child_detector.h5')
model2 = load_model('Bald_detector.h5')

#Initializing the GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Child and Bald Detector')
top.configure(background='#CDCDCD')

#Initializing the labels (1 for age and 1 for Sex)
label1 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
# label2 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label3 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
label4 = Label(top,background='#CDCDCD',font=('arial',15,"bold"))
sign_image = Label(top)


# Function to get the dominant color
def get_dominant_color(image):
    pixels = image.reshape(-1, image.shape[-1])
    dominant_color = mode(pixels, axis=0, keepdims=True).mode.flatten()
    return dominant_color


# Function to get the eye color
def get_eye_color(dominant_color):
    color_ranges = {
        'blue': ([90, 0, 0], [255, 95, 95]),
        'green': ([0, 90, 0], [95, 255, 95]),
        'brown': ([0, 0, 90], [95, 95, 255]),
        'hazel': ([40, 50, 0], [180, 160, 50]),
        'amber': ([40, 60, 80], [180, 150, 120]),
        'gray': ([40, 40, 40], [180, 180, 180]),
        'violet': ([100, 0, 100], [180, 95, 255]),
        'black': ([10, 10, 10], [40, 40, 40]),
        'red': ([120, 0, 0], [255, 40, 40])
    }

    for color, (lower, upper) in color_ranges.items():
        for color_val in dominant_color:
            if np.all(color_val >= lower) and np.all(color_val <= upper):
                return color

    return 'brown'


#Defining Detect function which detects the age and gender of the person in image using the model
def Detect(file_path):
    global label_packed

    # Load and preprocess the selected image
    img1 = Image.open(file_path)
    img1 = img1.resize((256, 256))  # Resize the image to match the model's input shape
    img_array = tf.keras.preprocessing.image.img_to_array(img1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    # Perform prediction
    result = model2.predict(img_array)
    prediction = result[0][0]  # Get the predicted value

    image = Image.open(file_path)
    image=image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    print(image.shape)

    

    # Detect eye color
    eye_color = detect_eye_color(file_path)
    # Display eye color
    label3.configure(foreground='#011638', text="Eye Color: " + eye_color)

    # sex_f=["Male","Female"]
    # image=np.array([image])/255
    result2 = model.predict(image)
    # age = int(np.round(pred[1][0]))
    # sex = int(np.round(pred[0][0]))
    if result2[0][0] > result2[0][1]:
        prediction_text2 = "Adult"
    else:
        prediction_text2 = "Child"
    
    label1.configure(foreground='#011638',text=prediction_text2)

    if prediction < 0.55:
        label4.configure(foreground='#011638',text="Predicted: Bald")
    else:
        label4.configure(foreground='#011638',text="Predicted: Not Bald")

    # if age<18:
    #     label1.configure(foreground='#011638',text="Predicted: Child")
    # else:
    #     label1.configure(foreground='#011638',text="Predicted: Not A Child")
    # print("Predicted age is "+str(age))
    # print("Predicted gender is "+sex_f[sex])
    # label1.configure(foreground='#011638',text=age)
    # label2.configure(foreground='#011638',text=sex_f[sex])

# Function to detect eye color
def detect_eye_color(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) > 0:
        (x, y, w, h) = eyes[0]
        eye_region = gray[y:y+h, x:x+w]
        eye_dominant_color = get_dominant_color(eye_region)
        eye_color = get_eye_color(eye_dominant_color)
        return eye_color
    else:
        return 'black'

#Defining Show_detect button function
def show_Detect_Button(file_path):
    Detect_b=Button(top,text='Detect Image',command=lambda: Detect(file_path),padx=1,pady=5)
    Detect_b.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
    Detect_b.place(relx=0.79,rely=0.46)

#Defining the upload image function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        # label2.configure(text='')
        show_Detect_Button(file_path)
    except Exception as e:
        messagebox.showinfo("Error", "Failed to open image: " + str(e))

upload = Button(top,text="Upload An Image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156',foreground='white',font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)

label1.pack(side="bottom",expand=True)
# label2.pack(side="bottom",expand=True)
label3.pack(side="bottom", expand=True)
label4.pack(side="bottom", expand=True)
heading = Label(top,text='Child And Eye Color Predictor',pady=20,font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

