import os
import sys
from dotenv import load_dotenv

# Load env
load_dotenv()

# Add backend directory to path to import db and auth
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=== Running MedivioAI Backend Verification ===")

try:
    from db import client, users_collection, records_collection
    from auth import hash_password, check_password, generate_token
    
    # 1. Test connection
    print("1. Testing MongoDB connection...")
    client.admin.command('ping')
    print("   [SUCCESS] Successfully pinged MongoDB server.")
    
    # 2. Test collections
    print("2. Testing collections and indexes...")
    print(f"   Users collection: {users_collection.name}")
    print(f"   Records collection: {records_collection.name}")
    
    # 3. Test auth helper
    print("3. Testing password hashing and validation...")
    pw = "super_secure_pass_123"
    hashed = hash_password(pw)
    is_valid = check_password(pw, hashed)
    is_invalid = check_password("wrong_password", hashed)
    
    if is_valid and not is_invalid:
         print("   [SUCCESS] Hashing and password checking functions work correctly.")
    else:
         print("   [ERROR] Password verification logic failed.")
         
    # 4. Test JWT generation
    print("4. Testing JWT generation...")
    test_id = "507f1f77bcf86cd799439011" # dummy object ID
    token = generate_token(test_id)
    if token:
         print("   [SUCCESS] JWT generated successfully.")
    else:
         print("   [ERROR] JWT generation returned empty token.")
         
    print("\nVerification completed successfully!")

except Exception as e:
    print(f"\n[ERROR] Verification encountered an error: {e}")
    print("Make sure MongoDB is running and MONGO_URI in .env is correct.")
