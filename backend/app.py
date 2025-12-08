import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
CORS(app)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = {
    "pneumonia": os.path.join(BASE_DIR, 'models', 'pneumonia', 'pneumonia_model_best.keras'),
    "tuberculosis": os.path.join(BASE_DIR, 'models', 'tuberculosis', 'tb_detection_model_best.h5'),
}

models = {}

def load_models():
    """Load available models into memory."""
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = keras.models.load_model(path)
            print(f"Loaded {name} model from {path}")
        except Exception as e:
            print(f"Error loading {name} model from {path}: {e}")

# Load models on startup
load_models()

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.get("/")
def index():
    """Root endpoint to avoid 404 and show API status."""
    return jsonify({
        "app": "MedivioAI Backend",
        "status": "running",
        "models_loaded": list(models.keys()),
        "endpoints": [
            {"method": "GET", "path": "/api/health"},
            {"method": "POST", "path": "/api/predict"}
        ]
    })

@app.get("/favicon.ico")
def favicon():
    """Return empty favicon to prevent 404 spam in logs when using a browser."""
    return ("", 204)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys())
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Select model (default to pneumonia)
        model_name = request.form.get('model', 'pneumonia').lower()
        if model_name not in models:
            return jsonify({
                'success': False,
                'error': f"Requested model '{model_name}' is not available",
                'available_models': list(models.keys()),
            }), 400
        model = models[model_name]
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        probability = float(prediction[0][0])

        # Determine class labels based on model
        if model_name == 'tuberculosis':
            positive_label = 'TUBERCULOSIS'
        else:
            positive_label = 'PNEUMONIA'

        if probability > 0.5:
            predicted_class = positive_label
            confidence = probability * 100
        else:
            predicted_class = 'NORMAL'
            confidence = (1 - probability) * 100

        return jsonify({
            'success': True,
            'model_used': model_name,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'probability': {
                'normal': round((1 - probability) * 100, 2),
                positive_label.lower(): round(probability * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)