# Required for Streamlit
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
# st.set_page_config(layout="wide")

# Required for classification
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2

# Global Variables for classification

# load trained model and cache it
@st.cache(allow_output_mutation=True)
def retrieve_model():
    model = load_model('model/Complex_PostZoomFix.h5')
    model.make_predict_function()
    return model

model = retrieve_model()

# Define minimum brightness as percent
min_brightness = 0.75

# Define list of candidate names
target_names = ['ramen', 'pizza', 'shawarma', 'spaghetti']

# First part of streamlit styling and functionality

# Title
st.title('Automated Write-In Resolution')
st.subheader('Please vote for your favourite food: ramen, pizza, shawarma, or spaghetti!')

# slider for min similarity threshold
min_sim = st.slider(label='Minimum similarity threshold:',
          min_value = 0.5,
          max_value = 1.0,
          value = 0.6,
          step = 0.05)
             
# Create a canvas component
canvas_result = st_canvas(
    stroke_width=5,
    stroke_color='#000000',
    background_color='#ffffff',
    background_image=None,
    height=100,
    width=700,
    key="canvas",
)

# Functions for classification

# preprocess image function
def preprocess(img):
    
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply gaussian blur to remove noise from image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred

# correct brightness function (for optically scanned votes)
def brightness(img, minimum_brightness=min_brightness):
    
    # extract measure of brightness and ratio compared to minimum
    cols, rows = img.shape
    brightness = np.sum(img) / (255 * cols * rows)
    ratio = brightness / minimum_brightness
   
    # if it's already bright enough, return it unchanged
    if ratio >= 1:
        return img

    # Otherwise, adjust brightness to get the target brightness
    return cv2.convertScaleAbs(img, alpha = 1 / ratio, beta = 0)

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
    
    return cnts, thresh

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

def predict_drawn(boxes, chars, image, model=model):

    # OCR the characters using our handwriting recognition model
    preds = model.predict(chars)
    
    # define the list of label names
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in alpha]
    
    # instantiate name string container 
    pred_name = ''
    
    # convert image back to color
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    
    # loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        
        # extract most likely label, probability
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]
        
        # add predicted character to pred_name container string
        pred_name += label
        
        # draw the prediction on the image
        cv2.rectangle(image, (x, y+50), (x + w, y + h + 50), (0, 255, 0), 2)
        cv2.putText(image, label, (x - 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    return image, pred_name

def predict_upload(boxes, chars, image, model=model):

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
        cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    
    return image, pred_name

# calculate the Levenshtein Distance between two strings
def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

# function to predict which candidate receives the vote
def result(pred_name, target_names=target_names, min_sim=min_sim):
    
    # instantiate results container as 0, indicating no good result
    result = 0
    
    # loop over target (registered writein) names
    for i, name in enumerate(target_names):
        
        # remove spaces, convert to uppercase
        name = name.upper().replace(' ','')
        
        # extract edit distance between result and target name, normalized by string length
        maxlen = max(len(pred_name), len(name))
        similarity = (maxlen - levenshteinDistance(pred_name, name)) / maxlen
        
        # if more than one has max, return result 1, indicating conflict
        if similarity == min_sim and result != 0:
            result = 1  
        
        # otherwise return minimum edit distance and winning name
        elif similarity > min_sim:
            result = target_names[i]
            min_sim = similarity     

    return result, min_sim

# container function for in-app drawing classification
def classify_drawn(image):
    
    # simple preprocessing on digitally drawn image
    image = image.astype("uint8")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnts, thresh = extract_contours(image)
    boxes, chars = extract_characters(cnts, image)
    
    # add space to image in order to correctly render prediction boxes
    im_padded = cv2.copyMakeBorder(image.copy(), 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255,255,255])
    
    res_img, pred_name = predict_drawn(boxes, chars, im_padded)
    res, min_sim = result(pred_name)
    
    return res_img, res, min_sim

# container function for upload classification
def classify_upload(image):
    blurred = preprocess(img_array)
    bright = brightness(blurred)
    cnts, thresh = extract_contours(bright)
    boxes, chars = extract_characters(cnts, blurred)
    res_img, pred_name = predict_upload(boxes, chars, image.copy())
    res, min_sim = result(pred_name)
    return res_img, res, min_sim

# Submission button for drawn
if st.button(label='Submit'):
    res_img, res, min_sim = classify_drawn(canvas_result.image_data)
    
    # instantiate results from csv
    results = pd.read_csv('data/results/results.csv', index_col=0)
    
    if res == 1:
        st.write("I can't resolve this to a candidate because there are two candidates with equal similarity to what was written. This is how the image was interpreted:")
        st.image(res_img)
        st.write('No votes were recorded.')
    elif res == 0:
        st.write("I can't resolve this to a candidate because no candidates were above the similarity threshold. This is how the image was interpreted:")
        st.image(res_img)
        st.write('No votes were recorded.')
    else:
        st.write(f'The similarity between what was written and the closest candidate is {round(min_sim*100,2)}%. This is how the image was interpreted:')
        st.image(res_img)
        st.write(f'One vote has been recorded for {res}.')
        results.loc[res] += 1
        results.to_csv('data/results/results.csv')
    
    st.bar_chart(results)
    
# Submission button for upload

st.write('OR upload an image')

img_file_buffer = st.file_uploader(label='Upload', type=["png", "jpg", "jpeg"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    res_img, res, min_sim = classify_upload(img_array)
    
    # instantiate results from csv
    results = pd.read_csv('data/results/results.csv', index_col=0)
    
    if res == 1:
        st.write("I can't resolve this to a candidate because there are two candidates with equal similarity to what was written. This is how the image was interpreted:")
        st.image(res_img)
        st.write('No votes were recorded.')
    elif res == 0:
        st.write("I can't resolve this to a candidate because no candidates were above the similarity threshold. This is how the image was interpreted:")
        st.image(res_img)
        st.write('No votes were recorded.')
    else:
        st.write(f'The similarity between what was written and the closest candidate is {round(min_sim*100,2)}%. This is how the image was interpreted:')
        st.image(res_img)
        st.write(f'One vote has been recorded for {res}.')
        results.loc[res] += 1
        results.to_csv('data/results/results.csv')
    
    st.bar_chart(results)
    
st.write('For more information, check out the github repo at https://github.com/monolith1/Handwritten-OCR-for-WriteIn-Resolution')