import cv2
import numpy as np


hue_ranges = {
    'red': ((0, 10), (170, 180)),
    'orange': (10,20),
    'yellow': (20,30),
    'greenish-yellow': (30,45),
    'green': (45, 75),
    'cyan-light blue':(75,90),
    'blue': (90, 130)
}


def get_predominant_color(hue):
    for color, ranges in hue_ranges.items():
        if isinstance(ranges[0], tuple):  # Handling red's wrap-around
            if (ranges[0][0] <= hue <= ranges[0][1]) or (ranges[1][0] <= hue <= ranges[1][1]):
                return color
        else:
            if ranges[0] <= hue <= ranges[1]:
                return color
    return None

# Pixel 2 pixel very slow

def adjust_image_pixelwise(img, new_h, new_s, new_v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Compute the scaling factors
    avg_h = np.mean(h)/180
    hue_scale = new_h / (avg_h + 1e-6)  # Prevent division by zero
    new_hues = np.clip(h * hue_scale,0,180).astype(np.uint8)  # Calculate new hues for all pixels

    # Apply hue scaling only if the predominant color remains unchanged
    for i in np.ndindex(h.shape):  # More efficient iteration over indices
        original_color = get_predominant_color(h[i])
        new_color = get_predominant_color(new_hues[i])
        if original_color == new_color:
            h[i] = new_hues[i]
    
    # Scale saturation and value
    avg_s = np.mean(s)/255
    saturation_scale = new_s / (avg_s + 1e-6)
    s = np.clip(s * saturation_scale, 0, 255).astype(np.uint8)
    
    avg_v = np.mean(v)/255
    value_scale = new_v / (avg_v + 1e-6)
    v = np.clip(v * value_scale, 0, 255).astype(np.uint8)

    # Merge and convert back to BGR
    hsv_adjusted = cv2.merge([h, s, v])
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return img_adjusted


# Image as a whole
def adjust_image(img, new_h, new_s, new_v):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split into the H, S, and V channels
    h, s, v = cv2.split(hsv)
    avg_h=np.mean(h)
    original_color = get_predominant_color(avg_h)
    normal_h = avg_h/180
    hue_scale = new_h/normal_h
            # Scale the hue value
            # Ensure the hue value wraps correctly
    new_hue = np.clip(h * hue_scale, 0, 180).astype(hsv.dtype)
    new_color = get_predominant_color(np.mean(new_hue))

    # Check if the predominant color remains the same
    if original_color == new_color:
        h = new_hue
    
    avg_s = (np.mean(s))/255
    saturation_scale = new_s/avg_s
    s = np.clip(s * saturation_scale, 0, 255).astype(hsv.dtype)
    avg_v = (np.mean(v))/255
    value_scale = new_v/avg_v
    v = np.clip(v * value_scale, 0, 255).astype(hsv.dtype)
    hsv_adjusted = cv2.merge([h, s, v])
    # Convert back from HSV to BGR
    img_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    return img_adjusted

def get_mean_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split into the H, S, and V channels
    h, s, v = cv2.split(hsv)
    avg_h=np.mean(h)/180
    avg_s=np.mean(s)/255
    avg_v=np.mean(v)/255
    return avg_h,avg_s,avg_v