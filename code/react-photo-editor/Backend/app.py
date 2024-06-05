from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
import base64


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense , Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
import shutil

import os
import csv
import cv2
# from your_ml_model import predict_images  # Replace with your ML model's prediction function


modelAH = './Ann/NewModels3D/A/HueModel.keras'
modelAS = './Ann/NewModels3D/A/SaturationModel.keras'
modelAv = './Ann/NewModels3D/A/ExposureModel.keras'

modelBH = './Ann/NewModels3D/B/HueModel.keras'
modelBS = './Ann/NewModels3D/B/SaturationModel.keras'
modelBv = './Ann/NewModels3D/B/ExposureModel.keras'

modelCH = './Ann/NewModels3D/C/HueModel.keras'
modelCS = './Ann/NewModels3D/C/SaturationModel.keras'
modelCv = './Ann/NewModels3D/C/ExposureModel.keras'


loaded_modelAH = load_model(modelAH)
loaded_modelAS = load_model(modelAS)
loaded_modelAV = load_model(modelAv)

loaded_modelBH = load_model(modelBH)
loaded_modelBS = load_model(modelBS)
loaded_modelBV = load_model(modelBv)

loaded_modelCH = load_model(modelCH)
loaded_modelCS = load_model(modelCS)
loaded_modelCV = load_model(modelCv)

def calculate_all_parameters_with_cv2(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    average_hue = np.mean(H)
    average_saturation = np.mean(S)
    average_exposure = np.mean(V)
    return (average_hue/180, average_saturation/255, average_exposure/255)


def PredictHSV_A(originalH,originalS,originalV):
  data = {'Hue': [originalH], 'Saturation': [originalS], 'Exposure': [originalV]}
  df = pd.DataFrame(data)
  y_predH = loaded_modelAH.predict(df)
  y_predS = loaded_modelAS.predict(df)
  y_predV = loaded_modelAV.predict(df)
  return y_predH,y_predS,y_predV

def PredictHSV_B(originalH,originalS,originalV):
  data = {'Hue': [originalH], 'Saturation': [originalS], 'Exposure': [originalV]}
  df = pd.DataFrame(data)
  y_predH = loaded_modelBH.predict(df)
  y_predS = loaded_modelBS.predict(df)
  y_predV = loaded_modelBV.predict(df)
  return y_predH,y_predS,y_predV

def PredictHSV_C(originalH,originalS,originalV):
  data = {'Hue': [originalH], 'Saturation': [originalS], 'Exposure': [originalV]}
  df = pd.DataFrame(data)
  y_predH = loaded_modelCH.predict(df)
  y_predS = loaded_modelCS.predict(df)
  y_predV = loaded_modelCV.predict(df)
  return y_predH,y_predS,y_predV


def adjust_image(img,new_hue, new_saturation,new_val ):
    # Convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split into the H, S, and V channels
    h, s, v = np.mean(hsv, axis=(0, 1))
    h_shift = (new_hue*179 - h)/3
    S_x = new_saturation*255 - s
    if S_x>0:
      s_shift =S_x/1.2
    else:
      s_shift = (S_x+30)/7
    
    V_x = new_val*255 - v 
    if V_x>0:
      v_shift = V_x/1.4
    else:
      v_shift = (V_x)/4.2
    # print( h_shift , s_shift , v_shift)
    
    # Adjust HSV values
    adjusted_hsv_image = hsv.astype("float32")
    adjusted_hsv_image[..., 0] += h_shift
    adjusted_hsv_image[..., 1] += s_shift
    adjusted_hsv_image[..., 2] += v_shift
    adjusted_hsv_image = np.clip(adjusted_hsv_image, 0, 255).astype("uint8")
    # Adjust values
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    
    return adjusted_image

def predict_images(files):
  adjust_images = []
  for i, file in enumerate(files):
    img = cv2.imread(file)
    
    originalH,originalS,originalV = calculate_all_parameters_with_cv2(img)
    NewH,NewS,NewV = PredictHSV_A(originalH,originalS,originalV)
    img_adjusted = adjust_image(img, new_hue=NewH, new_saturation=NewS, new_val=NewV)
    _, encoded_image = cv2.imencode('.jpg', img_adjusted)
    encoded_image = np.array(encoded_image).tobytes()
    adjust_images.append(encoded_image)
    
    NewH,NewS,NewV = PredictHSV_B(originalH,originalS,originalV)
    img_adjusted = adjust_image(img, new_hue=NewH, new_saturation=NewS, new_val=NewV)
    _, encoded_image = cv2.imencode('.jpg', img_adjusted)
    encoded_image = np.array(encoded_image).tobytes()
    adjust_images.append(encoded_image)
    
    NewH,NewS,NewV = PredictHSV_C(originalH,originalS,originalV)
    img_adjusted = adjust_image(img, new_hue=NewH, new_saturation=NewS, new_val=NewV)
    _, encoded_image = cv2.imencode('.jpg', img_adjusted)
    encoded_image = np.array(encoded_image).tobytes()
    adjust_images.append(encoded_image)
    
    
    
  encoded_images = [base64.b64encode(image).decode('utf-8') for image in adjust_images]
  return encoded_images


def cleanup_uploads():
    upload_folder = app.config['UPLOAD_FOLDER']
    if os.path.exists(upload_folder):
        shutil.rmtree(upload_folder)  # Remove the entire uploads folder and its contents


  
app = Flask(__name__)
CORS(app, origins=['http://localhost:5173'])
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_files():
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    # print(files)
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    file_paths = []
    for file in files:
        file_name = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
        file.save(file_path)
        file_paths.append(file_path)

    predicted_results = predict_images(file_paths)
    cleanup_uploads()
    return jsonify(predicted_results)

if __name__ == '__main__':
    app.run(debug=True)


