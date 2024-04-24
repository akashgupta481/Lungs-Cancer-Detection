from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os

app = Flask(__name__)

# Load the trained CNN model
cnn_model = load_model('/Users/akash/MCA Academics/Minor Project/Lung Cancer Detection Final/model.h5')  # Update with your model path

# Define function to preprocess image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Define route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the uploaded file temporarily
        file_path = 'temp_img.jpg'
        file.save(file_path)

        # Preprocess the image
        processed_img = preprocess_image(file_path)

        # Make prediction
        result = cnn_model.predict(processed_img)
        prediction = 'Normal' if result[0][0] == 1 else 'Cancer'

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
