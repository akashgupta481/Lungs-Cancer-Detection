
# Lung Cancer Detection

This project aims to detect lung cancer using a Convolutional Neural Network (CNN) model deployed with Flask. It includes a Jupyter notebook (`lung_cancer_detection.ipynb`) for model training and a Flask app (`app.py`) for making predictions. Additionally, an HTML template (`index.html`) is provided for the web interface.

## Model
- The model is trained using the notebook `lung_cancer_detection.ipynb`.
- It uses a CNN architecture to classify CT scan images into two categories: Normal and Cancerous.

## Flask App
- The Flask app (`app.py`) serves as the backend for making predictions.
- It loads the trained model and preprocesses images uploaded by users for prediction.
- Predictions are displayed on the web interface.

## Web Interface
- The web interface (`index.html`) allows users to upload CT scan images and receive predictions for lung cancer.
- It provides feedback on the prediction (Normal or Cancerous).

## Usage
- Clone the repository: `git clone <repository_url>`
- Install required dependencies: `pip install -r requirements.txt`
- Run the Flask app: `python app.py`
- Access the web interface in your browser at `http://localhost:5000`
- Make sure 'Model.h5' path is correct in 'webApp.py'

## Contact
For any questions or issues, please contact:
- Name: Akash Gupta
- Email: akashgupta0919@gmail.com

## Home Page Screenshot

<img width="1440" alt="Screenshot 2024-04-25 at 3 04 59 AM" src="https://github.com/akashgupta481/Lungs-Cancer-Detection/assets/126176668/7c7cb63f-2934-4680-9dd1-f31d2c65fc42">

## Detect Page Screenshot

<img width="1440" alt="Screenshot 2024-04-25 at 2 46 44 PM" src="https://github.com/akashgupta481/Lungs-Cancer-Detection/assets/126176668/2979f521-f387-442c-857e-de01ec97d1d4">

## Contact Page Screenshot

<img width="1440" alt="Screenshot 2024-04-25 at 2 47 12 PM" src="https://github.com/akashgupta481/Lungs-Cancer-Detection/assets/126176668/cba66d6b-a1cd-4ac5-bd31-3573553cce32">
