import numpy as np
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

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError


import os
import csv
import cv2



# test_folder = './Results/Raw'

modelH = './NewModels3D/A/HueModel.keras'
modelS = './NewModels3D/A/SaturationModel.keras'
modelv = './NewModels3D/A/ExposureModel.keras'
# csv_filePred = './NewModels3D/A/resultPred.csv'
# output_folder = './Results/predicted'

# CSV_raw = './Results/adobe5k_a.csv'



def calculate_all_parameters_with_cv2(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_img)
    average_hue = np.mean(H)
    average_saturation = np.mean(S)
    average_exposure = np.mean(V)
    return (average_hue/180, average_saturation/255, average_exposure/255)


#iterater through the images and extract the exposure, saturation and hue 
#make a list of the values and return the list

def extract_values(folder_path):
    exposure = []
    saturation = []
    hue = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        my_tuple = calculate_all_parameters_with_cv2(img)
        exposure.append(my_tuple[2])
        saturation.append(my_tuple[1])
        hue.append(my_tuple[0])
    return exposure, saturation, hue


raw_exposure, raw_saturation, raw_hue = extract_values(test_folder)


dfRaw = pd.DataFrame({
    'Raw Hue': raw_hue,
    'Raw Saturation': raw_saturation,
    'Raw Exposure': raw_exposure
})

dfRaw.to_csv(CSV_raw,index=False)





loaded_modelH = load_model(modelH)
loaded_modelS = load_model(modelS)
loaded_modelV = load_model(modelv)

y_predH = loaded_modelH.predict(X_scaled)
y_pred_dfH = pd.DataFrame(y_predH)

y_predS = loaded_modelS.predict(X_scaled)
y_pred_dfS = pd.DataFrame(y_predS)


y_predV = loaded_modelV.predict(X_scaled)
y_pred_dfV = pd.DataFrame(y_predV)


df_combined = pd.concat([y_pred_dfH, y_pred_dfS, y_pred_dfV], axis=1)
df_combined.columns = ['Hue', 'Saturation', 'Exposure']

df_combined.to_csv(csv_filePred, index=False,header=False)






def adjust_image(img, new_val, new_saturation, new_hue):
    # Convert from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Split into the H, S, and V channels
    h, s, v = np.mean(hsv, axis=(0, 1))

    # print 
    
    
    h_shift = (new_hue*179 - h)/3
    S_x = new_saturation*255 - s
    if S_x>0:
      s_shift =S_x/1.2
    else:
      s_shift = (S_x+30)/7
    # s_shift = (new_saturation*255 - s +20)/7
    
    V_x = new_val*255 - v 
    if V_x>0:
      v_shift = V_x/1.4
    else:
      v_shift = (V_x)/4.2
    
    # h_shift = 0
    # s_shift = 0
    # v_shift = 0   
    print( h_shift , s_shift , v_shift)
    
    # Adjust HSV values
    adjusted_hsv_image = hsv.astype("float32")
    adjusted_hsv_image[..., 0] += h_shift
    adjusted_hsv_image[..., 1] += s_shift
    adjusted_hsv_image[..., 2] += v_shift
    adjusted_hsv_image = np.clip(adjusted_hsv_image, 0, 255).astype("uint8")
    # Adjust values
    adjusted_image = cv2.cvtColor(adjusted_hsv_image, cv2.COLOR_HSV2BGR)
    
    return adjusted_image
  
  
  
  
  # test_folder = './SeprateTest/test'

os.makedirs(output_folder, exist_ok=True)

files = os.listdir(test_folder)

new_val = []
new_saturation = []
new_hue = []

with open(csv_filePred, mode='r') as csv_filePred:
    csv_reader = csv.reader(csv_filePred)
    for row in csv_reader:
        new_hue.append(float(row[0]))
        new_saturation.append(float(row[1]))
        new_val.append(float(row[2]))
        
        

for i, file in enumerate(files):
    img = cv2.imread(os.path.join(test_folder, file))
    print(file)
    img_adjusted = adjust_image(img, new_val[i], new_saturation[i], new_hue[i])
    cv2.imwrite(os.path.join(output_folder, file), img_adjusted)

