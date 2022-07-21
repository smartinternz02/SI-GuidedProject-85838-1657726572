# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 19:25:30 2022

@author: Vishwa_Tej
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 23:50:52 2022

@author: Vishwa_Tej
"""
import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

#from tensorflow.keras.utils import img_to_array


app = Flask(__name__)

# Load both the Vegetables and Fruit models

model= load_model("Vegetable.h5")
model1= load_model("fruit.h5")

#Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Prediction Page
@app.route('/prediction')
def prediction():
    return render_template('predict.html')

@app.route('/predict', methods =['POST'])

def predict():
    if request.method  == 'POST':
        # get the file from post request
        f = request.files['image']
        # Save the file ti ./uploads
        basepath=os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        img = image.load_img(file_path, target_size= (128, 128))
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        plant = request.form['plant']
        print(plant)
        
        if (plant == "vegetable"):
            preds = model.predict(x)
            preds = np.argmax(preds)
            print(preds)
            df = pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds]['caution'])
        else:
            preds = model1.predict(x)
            preds = np.argmax(preds)
            print(preds)
            df = pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds]['caution'])  
            
            
        return df.iloc[preds]['caution']
        
if __name__== "__main__":
    app.run(debug=False)