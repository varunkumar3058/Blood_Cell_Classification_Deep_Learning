import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Initialise the Flask app
app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('model.h5')

# Class indices for mapping 
index_to_class = {
    0: 'EOSINOPHIL',
    1: 'LYMPHOCYTE',
    2: 'MONOCYTE',
    3: 'NEUTROPHIL'
}

# Function to apply HSV masking
def apply_hsv_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([110, 50, 50])
    upper_purple = np.array([170, 255, 255])
    mask = cv2.inRange(hsv_image, lower_purple, upper_purple)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# Function to increase contrast using CLAHE
def increase_contrast(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    contrast_enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    return contrast_enhanced_image

# Function to sharpen the image
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# Function to preprocess the image for the model
def preprocess_image(image):
    # Step 1: Apply HSV mask
    image = apply_hsv_mask(image)
    
    # Step 2: Enhance contrast
    image = increase_contrast(image)
    
    # Step 3: Sharpen the image
    image = sharpen_image(image)
    
    # Step 4: Resize and normalise for model input
    image = cv2.resize(image, (224, 224)) 
    image = img_to_array(image)          
    image = np.expand_dims(image, axis=0) 
    image = image / 255.0                  
    
    return image

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read the image using OpenCV
        image = cv2.imread(filepath)

        # Preprocess the image for prediction
        preprocessed_image = preprocess_image(image)

        # Predict the class of the image
        prediction = model.predict(preprocessed_image)[0]  

        # Debug the result
        print(model.predict(preprocessed_image))
        rounded_prediction = np.round(prediction, 3)
        print(f"Raw Prediction Array: {rounded_prediction}") 

        # Get the index of the highest confidence prediction
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Map the predicted index to the class label
        predicted_class = index_to_class[predicted_index]

        # Print more debugging info
        print(f"Preprocessed Image Shape: {preprocessed_image.shape}")
        print(f"Predicted Index: {predicted_index}")
        print(f"Confidence: {confidence}")
        print(f"Predicted Class: {predicted_class}")

        # Redirect to the result page with prediction results
        return redirect(url_for('display_result', filename=filename, prediction=predicted_class, confidence=confidence))

    return redirect(request.url)

# Route to display the prediction result
@app.route('/result/<filename>')
def display_result(filename):
    prediction = request.args.get('prediction')
    confidence = request.args.get('confidence')
    return render_template('result.html', filename=filename, prediction=prediction, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)