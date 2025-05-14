# config.py

import os
from dotenv import load_dotenv
from supabase.client import create_client, Client
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# API configuration
TWEETS_API_URL = os.getenv("TWEETS_API_URL", "https://flyfish-sever.app.n8n.cloud/webhook/get-data")

# Initialize Supabase client
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI embeddings
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Print configuration information (for debugging)
def print_config():
    print("➜ SUPABASE_URL:", SUPABASE_URL)
    print("➜ SUPABASE_KEY loaded?:", bool(SUPABASE_KEY))
    print("➜ TWEETS_API_URL:", TWEETS_API_URL)