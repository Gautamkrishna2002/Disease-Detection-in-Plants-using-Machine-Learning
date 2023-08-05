import os
from flask import Flask, redirect, render_template, request
from PIL import Image

import numpy as np

import pandas as pd
from tensorflow import keras
from keras.models import load_model
import cv2




BASE_PATH = os.getcwd()
APP_PATH = os.path.join(BASE_PATH, 'Plant-Disease-Detection\Flask Deployed App')

MODEL_PATH = os.path.join(APP_PATH, 'plant_disease.model')

UPLOAD_PATH = os.path.join(BASE_PATH, 'uploaded_images')

DISEASE_INFO_PATH = os.path.join(APP_PATH, 'disease_info.csv')
SUPPLEMENT_INFO_PATH = os.path.join(APP_PATH, 'supplement_info.csv')

disease_info = pd.read_csv(DISEASE_INFO_PATH , encoding='cp1252')
supplement_info = pd.read_csv(SUPPLEMENT_INFO_PATH,encoding='cp1252')


model = keras.models.load_model(MODEL_PATH)


def prediction(image_path):
    
    testing = cv2.imread(image_path)
    resized = cv2.resize(testing, (100, 100))

    normalized = resized/255.0
    reshaped = np.reshape(normalized, (1, 100, 100, 3))
    result = model.predict(reshaped)

    R = np.argmax(result)
    return R



app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join(UPLOAD_PATH, filename)
        print(file_path)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('submit.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
