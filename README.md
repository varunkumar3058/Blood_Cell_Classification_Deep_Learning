# Blood Cell Classification Using Deep Learning

## About the Project

I developed this project as part of my MSc Computer Science studies, focusing on the classification of blood cell images using deep learning.

The main aim was to explore how different deep learning models perform when classifying microscopic blood cell images. I developed and compared three models: a custom CNN, DenseNet121 and InceptionV3.

I also built a Flask web application that allows users to upload a blood cell image and view the predicted cell type along with its confidence score.

## Dataset

I used the Blood Cell Images dataset available on Kaggle.

**Dataset link:**  
https://www.kaggle.com/datasets/paultimothymooney/blood-cells

The dataset contains images of four types of white blood cells:

- Eosinophil
- Lymphocyte
- Monocyte
- Neutrophil

## Deep Learning Models

I experimented with and compared the following models:

- Custom Convolutional Neural Network (CNN)
- DenseNet121
- InceptionV3

## Model Results

Here is a comparison of the accuracy achieved by the three models:

| Model | Accuracy |
|---|---:|
| Custom CNN | 90.32% |
| DenseNet121 | 84.12% |
| InceptionV3 | 97.98% |

InceptionV3 performed best among the three models in my evaluation.

## Image Preprocessing

Before training and testing, I worked with several image preprocessing techniques:

- HSV colour masking
- Contrast enhancement using CLAHE
- Image sharpening
- Resizing images to 224 × 224 pixels
- Image normalisation

## Web Application

I also developed a Flask application to make the trained model easier to use. Users can upload an image through the web interface and receive a predicted blood cell category with a confidence score.

## Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Flask
- HTML and CSS

## Running the Application

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   ```

2. Navigate to the application folder:

   ```bash
   cd Blood_Cell_Classification_Web_App
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Flask application:

   ```bash
   python app.py
   ```

5. Open `http://127.0.0.1:5000/` in your browser.

## What I Learned

Working on this project gave me practical experience in deep learning, computer vision, image preprocessing, transfer learning and model evaluation. It also helped me understand how to integrate a trained machine learning model into a web application using Flask.