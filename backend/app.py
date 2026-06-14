import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import base64
import datetime
import threading
import jwt
from bson import ObjectId
from db import users_collection, records_collection
from auth import hash_password, check_password, generate_token, token_required

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Disease detection models
MODEL_PATHS = {
    "pneumonia": os.path.join(BASE_DIR, 'models', 'pneumonia', 'pneumonia_model_best.keras'),
    "tuberculosis": os.path.join(BASE_DIR, 'models', 'tuberculosis', 'tb_detection_model_best.h5'),
}

models = {}
models_ready = False
models_loading = False
models_load_error = None


def load_models():
    """Load available disease models into memory."""
    global models_ready, models_loading, models_load_error

    if models_ready or models_loading:
        return

    models_loading = True

    try:
        from tensorflow import keras
        keras.utils.disable_interactive_logging()
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        # Load disease-specific models
        for name, path in MODEL_PATHS.items():
            try:
                models[name] = keras.models.load_model(path)
            except Exception as e:
                logging.warning("Error loading %s model from %s: %s", name, path, e)

        models_ready = len(models) > 0
        if not models_ready:
            models_load_error = 'No models could be loaded'
    except Exception as e:
        models_load_error = str(e)
        logging.error("Failed to load models: %s", e)
    finally:
        models_loading = False


def start_model_loading():
    """Load models in a background thread so the API can start quickly."""
    threading.Thread(target=load_models, daemon=True).start()


# Start loading models without blocking app startup.
start_model_loading()


def preprocess_image(image, size=(224, 224)):
    """Preprocess image for model prediction."""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize to desired input size
    image = image.resize(size)

    # Convert to array and normalize
    img_array = np.array(image).astype("float32") / 255.0

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
            {"method": "POST", "path": "/api/predict"},
            {"method": "POST", "path": "/api/auth/register"},
            {"method": "POST", "path": "/api/auth/login"},
            {"method": "GET", "path": "/api/auth/profile"},
            {"method": "GET", "path": "/api/records"}
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

# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        name = data.get('name', '').strip()
        age = data.get('age')
        gender = data.get('gender', '').strip()
        
        if not email or not password or not name:
            return jsonify({
                'success': False,
                'error': 'Email, password, and name are required'
            }), 400
            
        # Check if user already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return jsonify({
                'success': False,
                'error': 'User with this email already exists'
            }), 400
            
        # Hash password and save
        hashed_password = hash_password(password)
        new_user = {
            'email': email,
            'password': hashed_password,
            'name': name,
            'age': age,
            'gender': gender,
            'created_at': datetime.datetime.utcnow()
        }
        
        result = users_collection.insert_one(new_user)
        token = generate_token(result.inserted_id)
        
        return jsonify({
            'success': True,
            'message': 'Account registered successfully',
            'token': token,
            'user': {
                'id': str(result.inserted_id),
                'email': email,
                'name': name,
                'age': age,
                'gender': gender
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json or {}
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({
                'success': False,
                'error': 'Email and password are required'
            }), 400
            
        user = users_collection.find_one({'email': email})
        if not user or not check_password(password, user['password']):
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401
            
        token = generate_token(user['_id'])
        
        return jsonify({
            'success': True,
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'age': user.get('age'),
                'gender': user.get('gender')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/profile', methods=['GET'])
@token_required
def get_profile():
    try:
        user = g.current_user
        return jsonify({
            'success': True,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'name': user['name'],
                'age': user.get('age'),
                'gender': user.get('gender'),
                'created_at': user.get('created_at').isoformat() if isinstance(user.get('created_at'), datetime.datetime) else user.get('created_at')
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==========================================
# MEDICAL RECORDS ENDPOINTS
# ==========================================

@app.route('/api/records', methods=['GET'])
@token_required
def get_records():
    try:
        user_id = g.current_user['_id']
        cursor = records_collection.find({'user_id': user_id}).sort('timestamp', -1)
        records = []
        for r in cursor:
            records.append({
                'id': str(r['_id']),
                'model_used': r['model_used'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'probability': r.get('probability', {}),
                'timestamp': r['timestamp'].isoformat() if isinstance(r['timestamp'], datetime.datetime) else r['timestamp'],
                'filename': r.get('filename', 'Unknown'),
                'image_preview': r.get('image_preview') # Base64 thumbnail
            })
            
        return jsonify({
            'success': True,
            'records': records
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        if not models_ready and not models_loading and not models:
            load_models()

        if not models:
            return jsonify({
                'success': False,
                'error': models_load_error or 'Models are still loading. Please try again in a moment.'
            }), 503

        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        file = request.files['image']
        image_bytes = file.read()

        # Open image once and reuse
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run disease-specific model (pneumonia / tuberculosis)
        # Select disease model (default to pneumonia)
        model_name = request.form.get('model', 'pneumonia').lower()
        if model_name not in models:
            return jsonify({
                'success': False,
                'error': f"Requested model '{model_name}' is not available",
                'available_models': list(models.keys()),
            }), 400
        model = models[model_name]

        # Reuse the same image object but preprocess for disease model input size (224x224)
        disease_input = preprocess_image(image, size=(224, 224))

        # Make prediction
        prediction = model.predict(disease_input, verbose=0)
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

        # Check if user is authenticated (optional)
        current_user_id = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
                try:
                    from auth import JWT_SECRET
                    payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                    current_user_id = payload['sub']
                except Exception as e:
                    logging.debug("Optional token authentication failed in predict: %s", e)

        # Save record if authenticated user
        if current_user_id:
            try:
                # Generate preview
                image_preview = None
                try:
                    thumb = image.copy()
                    thumb.thumbnail((150, 150))
                    buffered = io.BytesIO()
                    if thumb.mode != 'RGB':
                        thumb = thumb.convert('RGB')
                    thumb.save(buffered, format="JPEG", quality=70)
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    image_preview = f"data:image/jpeg;base64,{img_str}"
                except Exception as thumb_err:
                    logging.debug("Error creating thumbnail: %s", thumb_err)

                record = {
                    'user_id': ObjectId(current_user_id),
                    'model_used': model_name,
                    'prediction': predicted_class,
                    'confidence': round(confidence, 2),
                    'probability': {
                        'normal': round((1 - probability) * 100, 2),
                        positive_label.lower(): round(probability * 100, 2)
                    },
                    'timestamp': datetime.datetime.utcnow(),
                    'filename': file.filename,
                    'image_preview': image_preview
                }
                records_collection.insert_one(record)
            except Exception as save_err:
                logging.warning("Failed to save prediction record: %s", save_err)

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
    app.run(debug=True, use_reloader=False, threaded=True, port=5000)