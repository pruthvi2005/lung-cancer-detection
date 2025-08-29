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
    # Check if model file exists locally, if not download from GitHub releases
    import os
    if not os.path.exists(MODEL_PATH):
        print("* Model not found locally, downloading from GitHub...")
        import urllib.request
        model_url = "https://github.com/pruthvi2005/lung-cancer-detection/releases/download/v1.0/lung_cancer_model.h5"
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        print("* Model downloaded successfully")
    
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment platforms."""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': model is not None,
            'classes_loaded': class_names is not None,
            'timestamp': str(tf.timestamp())
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        # Check if model and class names are loaded
        if model is None:
            return jsonify({'error': 'AI model is not loaded. Please try again later.'}), 500
        
        if class_names is None:
            return jsonify({'error': 'Class indices not loaded. Please try again later.'}), 500

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No image file provided. Please select an image.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected. Please choose an image file.'}), 400

        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, JPEG, GIF, BMP, or WEBP images.'}), 400

        # Process the image
        try:
            # Read image bytes
            img_bytes = file.read()
            
            # Validate file size (max 10MB)
            if len(img_bytes) > 10 * 1024 * 1024:
                return jsonify({'error': 'File too large. Please upload an image smaller than 10MB.'}), 400
            
            if len(img_bytes) == 0:
                return jsonify({'error': 'Empty file. Please upload a valid image.'}), 400
            
            # Preprocess the image
            processed_image = preprocess_image(img_bytes)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            
            # Validate prediction output
            if prediction is None or len(prediction) == 0:
                return jsonify({'error': 'Model prediction failed. Please try again.'}), 500
            
            # Get the predicted class index
            predicted_class_index = np.argmax(prediction[0])
            
            # Validate class index
            if predicted_class_index not in class_names:
                return jsonify({'error': 'Invalid prediction result. Please try again.'}), 500
            
            # Get the class name
            predicted_class_name = class_names.get(predicted_class_index, "Unknown")
            
            # Get the confidence score
            confidence = float(prediction[0][predicted_class_index])
            
            # Validate confidence
            if confidence < 0 or confidence > 1:
                confidence = max(0, min(1, confidence))  # Clamp between 0 and 1

            return jsonify({
                'prediction': predicted_class_name.replace('_', ' ').title(),
                'confidence': f'{confidence:.2%}',
                'status': 'success'
            })
            
        except Exception as img_error:
            print(f"Image processing error: {img_error}")
            return jsonify({'error': 'Failed to process image. Please ensure it\'s a valid image file.'}), 500
            
    except Exception as e:
        print(f"General prediction error: {e}")
        return jsonify({'error': 'Service temporarily unavailable. Please try again later.'}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
