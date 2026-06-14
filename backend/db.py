import os
import logging
from pymongo import MongoClient
from dotenv import load_dotenv
from bson import ObjectId
from copy import deepcopy

# Load environment variables from .env file
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/medivioai")

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


class InMemoryInsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class InMemoryCursor:
    def __init__(self, documents):
        self._documents = documents

    def sort(self, key, direction):
        reverse = direction == -1
        self._documents.sort(key=lambda doc: doc.get(key), reverse=reverse)
        return self

    def __iter__(self):
        return iter(self._documents)


class InMemoryCollection:
    def __init__(self, name):
        self.name = name
        self._documents = []

    def create_index(self, *args, **kwargs):
        return None

    def _matches(self, document, query):
        for key, value in query.items():
            if document.get(key) != value:
                return False
        return True

    def find_one(self, query):
        for document in self._documents:
            if self._matches(document, query):
                return deepcopy(document)
        return None

    def insert_one(self, document):
        stored = deepcopy(document)
        if "_id" not in stored:
            stored["_id"] = ObjectId()
        self._documents.append(stored)
        return InMemoryInsertResult(stored["_id"])

    def find(self, query):
        matches = [deepcopy(document) for document in self._documents if self._matches(document, query)]
        return InMemoryCursor(matches)

# Initialize Mongo Client
use_in_memory_db = False
try:
    client = MongoClient(MONGO_URI)
    db = client.get_default_database(default="medivioai")
    
    # Ping database to verify connection
    client.admin.command('ping')
except Exception as e:
    logging.warning("Error connecting to MongoDB: %s", e)
    use_in_memory_db = True
    client = None
    db = None

if use_in_memory_db:
    logging.warning("Using in-memory database fallback. Data will not persist after restart.")
    users_collection = InMemoryCollection("users")
    records_collection = InMemoryCollection("records")
else:
    # Collections
    users_collection = db["users"]
    records_collection = db["records"]

# Setup indexes
try:
    users_collection.create_index("email", unique=True)
except Exception as e:
    logging.warning("Error creating database indexes: %s", e)
