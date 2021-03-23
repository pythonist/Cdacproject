from __future__ import division, print_function
from flask import Flask, render_template

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__,template_folder='templates')

model = load_model('model_pneum.h5')

model1= load_model('E:/Combination/models/model.h5')     

lesion_classes_dict = {
     0:'Melanocytic nevi',
     1:'Melanoma',
     2:'Benign keratosis-like lesions ',
     3:'Basal cell carcinoma',
     4:'Actinic keratoses',
     5:'Vascular lesions',
     6:'Dermatofibroma'
}

def model_predict(img_path, model):
    
    img = image.load_img(img_path, target_size=(200, 200),color_mode='grayscale') #target_size must agree with what the trained model expects!!

    # Preprocessing the image
    img = image.img_to_array(img)
    resized_arr=cv2.resize(img,(200,200))
    resized_arr=resized_arr.reshape(-1,200,200,1)

   
    preds = model.predict(resized_arr)
    return preds

def model_predict1(img_path, model1):
    img = image.load_img(img_path, target_size=(128,128,3))
  
    #img = np.asarray(pil_image.open('img').resize((120,120)))
    #x = np.asarray(img.tolist())

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    spreds = model1.predict(x.reshape(1,128,128,3))
    return spreds

@app.route('/')
def home():
   return render_template('home.html')

@app.route('/pnum_page', methods = ['GET','POST'])
def pnum_page():
    title = 'Second page'
    return render_template('pindex.html', title=title)

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = ' Patient Normal'
        str2 = ' Patient has Pneumonia'
        if (preds[0] == 1.):
            return str1
        else:
            return str2
   # return None


@app.route('/skin_page', methods = ['GET','POST'])
def skin_page():
    title = 'Skin page'
    return render_template('index1.html', title=title)

@app.route('/predict-skin', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        spreds = model_predict1(file_path , model1)
        
        top3 = np.argsort(spreds[0])[:-4:-1]
        result =[]
        for i in range(2):
           a = ("{}".format(lesion_classes_dict[top3[i]]))
       
        result.append(a) 
        result = str(result)
        # Process your result for human
        #pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        #pr = lesion_classes_dict[pred_class[0]]
        #result =str(pr)         
        return result
    return None

if __name__ == '__main__':
   app.run()
