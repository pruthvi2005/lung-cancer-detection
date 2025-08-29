# app.py
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import json

app = Flask(__name__)

# --- Load the trained model and class indices ---
MODEL_PATH = 'lung_cancer_model.h5'
CLASS_INDICES_PATH = 'class_indices.json'

# Load the model
try:
    model = load_model(MODEL_PATH)
    print("* Model loaded successfully")
except Exception as e:
    print(f"* Error loading model: {e}")
    model = None

# Load the class indices
try:
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
        # Invert the dictionary to map index to class name
        class_names = {v: k for k, v in class_indices.items()}
    print("* Class indices loaded successfully")
except Exception as e:
    print(f"* Error loading class indices: {e}")
    class_names = None


# --- Preprocessing function ---
def preprocess_image(image_bytes):
    """
    Preprocesses the uploaded image to be suitable for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    if model is None or class_names is None:
        return jsonify({'error': 'Model or class indices not loaded properly.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Read image bytes
            img_bytes = file.read()
            # Preprocess the image
            processed_image = preprocess_image(img_bytes)
            # Make prediction
            prediction = model.predict(processed_image)
            # Get the predicted class index
            predicted_class_index = np.argmax(prediction[0])
            # Get the class name
            predicted_class_name = class_names.get(predicted_class_index, "Unknown")
            # Get the confidence score
            confidence = float(prediction[0][predicted_class_index])

            return jsonify({
                'prediction': predicted_class_name.replace('_', ' ').title(),
                'confidence': f'{confidence:.2%}'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing file: {e}'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500


if __name__ == '__main__':
    app.run(debug=True)
