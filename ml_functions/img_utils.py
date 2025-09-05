import cv2
import numpy as np
import imutils

def find_receipt_contours(image, 
                          lower_white = np.array([0, 0, 120]), 
                          upper_white = np.array([255, 40, 255])):
    resized_height = 700 
    resized_image = imutils.resize(image, height=resized_height)
    scale_factor = image.shape[0] / resized_image.shape[0]
    
    hsv = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)    
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_receipts = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10000:
          found_receipts.append(c)
        
    return found_receipts, scale_factor

def crop_receipts(image, contours, scale_factor):
    cropped_images = []
    if contours is not None:
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            x_orig, y_orig = int(x * scale_factor), int(y * scale_factor)
            w_orig, h_orig = int(w * scale_factor), int(h * scale_factor)
            
            cropped = image[y_orig:y_orig+h_orig, x_orig:x_orig+w_orig]
            cropped_images.append(cropped)
    return cropped_images