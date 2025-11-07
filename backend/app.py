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

# Load model
MODEL_PATH = 'models/pneumonia_model_best.keras'
model = None

def load_model():
    global model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Load model on startup
load_model()

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
        "model_loaded": model is not None,
        "endpoints": [
            {"method": "GET", "path": "/api/health"},
            {"method": "POST", "path": "/api/predict"},
            {"method": "POST", "path": "/api/batch-predict"}
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
        'model_loaded': model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
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
        
        # Determine class
        if probability > 0.5:
            predicted_class = 'PNEUMONIA'
            confidence = probability * 100
        else:
            predicted_class = 'NORMAL'
            confidence = (1 - probability) * 100
        
        return jsonify({
            'success': True,
            'prediction': predicted_class,
            'confidence': round(confidence, 2),
            'probability': {
                'normal': round((1 - probability) * 100, 2),
                'pneumonia': round(probability * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        files = request.files.getlist('images')
        
        if not files:
            return jsonify({
                'success': False,
                'error': 'No images provided'
            }), 400
        
        results = []
        
        for file in files:
            try:
                image = Image.open(io.BytesIO(file.read()))
                processed_image = preprocess_image(image)
                
                prediction = model.predict(processed_image, verbose=0)
                probability = float(prediction[0][0])
                
                if probability > 0.5:
                    predicted_class = 'PNEUMONIA'
                    confidence = probability * 100
                else:
                    predicted_class = 'NORMAL'
                    confidence = (1 - probability) * 100
                
                results.append({
                    'filename': file.filename,
                    'prediction': predicted_class,
                    'confidence': round(confidence, 2)
                })
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)