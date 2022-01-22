# Required for Streamlit
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
st.set_page_config(layout="wide")

# Required for classification
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
import matplotlib.pyplot as plt

# Global Variables for classification

# load trained model
model = load_model('model/Complex.h5')

# Define list of candidate names
target_names = ['Mike Tyson', 'Kanye West', 'Tomoko Shinoda', 'Matt Coffey']

# First part of streamlit styling and functionality

# Title
st.title('Cast Your Vote')
st.header('Mike Tyson, Kanye West, Tomoko Shinoda, or Matt Coffey')

# Create a canvas component
canvas_result = st_canvas(
    stroke_width=7,
    stroke_color='#000000',
    background_color='#ffffff',
    background_image=None,
    height=200,
    width=1000,
    key="canvas",
)

# slider for max edit distance
max_ed = st.slider(label='Maximum allowed edit distance from closest candidate:',
          min_value = 0,
          max_value = 15,
          value = 8,
          step = 1)

# Functions for classification

# edge detection and contour extraction
def extract_contours(image):
    
    # Canny edge detection - use instead of otsu's thresholding for more exact but more segmented contours
    # high_thresh, thresh_im = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # lowThresh = 0.5*high_thresh
    # thresh = cv2.Canny(blurred, lowThresh, high_thresh)

    # apply otsu's thresholding - can be used instead of canny for less jagged contours
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # adaptive gaussian thresholding - something to try
    # thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    
    # extract contours from the binary image
    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort contours from left to right
    cnts = sort_contours(cnts, method="left-to-right")[0]
    
    return cnts

# extract formatted images of each character
def extract_characters(cnts, image):
    
    # initialize the list of boxes and associated characters
    chars = []
    
    # loop over the contours
    for i, c in enumerate(cnts):
        
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # filter out very large and very small boxes
        if ((image.shape[1] * 0.01) <=  w <= (image.shape[1] * 0.7)) and ((image.shape[0] * 0.03)<= h <= (image.shape[0] * 0.7)):

            # extract the character and threshold as white on black and retrieve dimensions
            roi = image[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # if the width is greater than the height, resize along the width dimension
            if tW > tH:
                thresh = imutils.resize(thresh, width=28)
            else:
                thresh = imutils.resize(thresh, height=28)

            # retrieve resized dims and determine if padding is necessary
            (tH, tW) = thresh.shape
            dX = int(max(0, 36 - tW) / 2.0)
            dY = int(max(0, 36 - tH) / 2.0)

            # pad the image and force 28x28 dimensions
            padded = cv2.copyMakeBorder(thresh,
                                        top=dY,
                                        bottom=dY,
                                        left=dX,
                                        right=dX,
                                        borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))
            padded = cv2.resize(padded, (28, 28))

            # perform final preprocessing for OCR
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # update our list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))
        
    # extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")
    
    return boxes, chars

def predict(boxes, chars, image, model=model):

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)
    
    # define the list of label names
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in alpha]
    
    # instantiate name string container 
    pred_name = ''
    
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        
        # extract most likely label, probability
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        
        # add predicted character to pred_name container string
        pred_name += label
        
        # draw the prediction on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    return image, pred_name

# function to calculate edit distance between strings
def edit_distance(s1, s2):
    m=len(s1)+1
    n=len(s2)+1

    tbl = {}
    for i in range(m): tbl[i,0]=i
    for j in range(n): tbl[0,j]=j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            tbl[i,j] = min(tbl[i, j-1]+1, tbl[i-1, j]+1, tbl[i-1, j-1]+cost)

    return tbl[i,j]

# function to predict which candidate receives the vote
def result(pred_name, target_names=target_names, max_ed = max_ed):
    result = ''
    for i, name in enumerate(target_names):
        # remove spaces, convert to uppercase
        name = name.upper().replace(' ','')
        ed = edit_distance(pred_name, name)
        if ed <= max_ed:
            result = target_names[i]
            max_ed = ed
    return result, max_ed

# container function for the entire results generation process
def classify(image):
    
    image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnts = extract_contours(image)
    boxes, chars = extract_characters(cnts, image)
    res_img, pred_name = predict(boxes, chars, image.copy())
    res, max_ed = result(pred_name)
    
    return res_img, res, max_ed

# Submission button
if st.button(label='Submit'):
    res_img, res, max_ed = classify(canvas_result.image_data)
    
    st.subheader(f'The edit distance between what was written and the closest candidate was {max_ed}. The image was interpreted like:')
    st.image(res_img)
    if res == '':
        st.subheader("I can't resolve this to a candidate.")
    else:
        st.subheader(f'One vote for {res}.')