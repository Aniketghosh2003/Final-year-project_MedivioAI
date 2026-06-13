import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/medivioai")

print(f"Connecting to MongoDB at: {MONGO_URI}")

# Initialize Mongo Client
try:
    client = MongoClient(MONGO_URI)
    db = client.get_default_database(default="medivioai")
    
    # Ping database to verify connection
    client.admin.command('ping')
    print("MongoDB connection established successfully.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # Create client without validation so app doesn't crash on startup if Mongo is down
    client = MongoClient(MONGO_URI)
    db = client["medivioai"]

# Collections
users_collection = db["users"]
records_collection = db["records"]

# Setup indexes
try:
    users_collection.create_index("email", unique=True)
    print("Database indexes created successfully.")
except Exception as e:
    print(f"Error creating database indexes: {e}")
