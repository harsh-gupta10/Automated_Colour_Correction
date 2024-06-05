import cv2
import numpy as np

# This will form the value for exposure
def calculate_exposure_with_cv2(img):
    # Convert the image from BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract the Value channel
    _, _, V = cv2.split(hsv_img)
    # Calculate average brightness/exposure
    average_exposure = np.mean(V)
    return (average_exposure/255)


def calculate_saturation_with_cv2(img):
    # Convert the image from BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract the Saturation channel
    _, S, _ = cv2.split(hsv_img)
    # Calculate average saturation
    average_saturation = np.mean(S)
    return (average_saturation/255)

def calculate_hue_with_cv2(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Extract the Hue channel
    H, _, _ = cv2.split(hsv_img)
    # Calculate average hue
    average_hue = np.mean(H)
    return (average_hue/180)

def extract_hsv(img):
    hsv_img= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv_img)
    average_H=np.mean(H)/180
    average_S=np.mean(S)/255
    average_V=np.mean(V)/255
    return [average_H,average_S,average_V]




