import os
import datetime
from functools import wraps
from flask import request, jsonify, g
import jwt
import bcrypt
from bson import ObjectId
from db import users_collection

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkeymedivioai")

def hash_password(password):
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def check_password(password, hashed_password):
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def generate_token(user_id, expires_in_days=7):
    """Generate a JWT token for a given user ID."""
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=expires_in_days),
        'iat': datetime.datetime.utcnow(),
        'sub': str(user_id)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def token_required(f):
    """Decorator to require a valid JWT token in the Authorization header."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(" ")[1]
        
        if not token:
            return jsonify({
                'success': False,
                'message': 'Access denied. Token is missing.'
            }), 401
            
        try:
            # Decode token
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            user_id = payload['sub']
            
            # Fetch user from database
            user = users_collection.find_one({"_id": ObjectId(user_id)})
            if not user:
                return jsonify({
                    'success': False,
                    'message': 'User not found.'
                }), 401
                
            # Store user in Flask glob context
            g.current_user = user
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'success': False,
                'message': 'Session expired. Please log in again.'
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'success': False,
                'message': 'Invalid session token. Please log in again.'
            }), 401
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Authentication error: {str(e)}'
            }), 401
            
        return f(*args, **kwargs)
        
    return decorated
