from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set up MongoDB connection using the URI from the environment variables
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MONGO_URI is not set in the .env file")

client = MongoClient(mongo_uri)
db = client["sleep_quality_db"]
