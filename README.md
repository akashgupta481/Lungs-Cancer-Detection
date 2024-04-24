
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

## Contact
For any questions or issues, please contact:
- Name: [Your Name]
- Email: [Your Email]

## License
This project is licensed under the [MIT License](LICENSE).
