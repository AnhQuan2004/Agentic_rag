# import basics
import os
import json
from dotenv import load_dotenv
import time
from tqdm import tqdm
import re
import requests

# import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Function to clean text of problematic characters
def clean_text(text):
    if text is None:
        return ""
    # Remove null bytes and other problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace other potentially problematic Unicode characters
    text = text.replace('\\u0000', '')
    return text

# Clean document content
def clean_documents(docs):
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        # Clean metadata values that are strings
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                doc.metadata[key] = clean_text(value)
    return docs

# Function to fetch tweet data from API
def fetch_tweet_data(address="0x000123456789"):
    url = "https://n8n.vbi-server.com/webhook/get-data"
    params = {
        "address": address
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for 4XX/5XX errors
        data = response.json()
        print(f"📊 Loaded {len(data['data'])} tweets from API")
        return data['data']
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from API: {str(e)}")
        return []
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error parsing API response: {str(e)}")
        return []

# Function to check if a tweet ID already exists in the database
def check_tweet_exists(tweet_id):
    try:
        # Query to check if a record with the given ID exists in tweet_id column
        response = supabase.table("tweets").select("id").eq("tweet_id", str(tweet_id)).execute()
        # If any records are returned, the tweet exists
        return len(response.data) > 0
    except Exception as e:
        print(f"❌ Error checking for existing tweet: {str(e)}")
        # If there's an error in checking, we'll assume it doesn't exist to be safe
        return False

# Function to insert tweets directly into Supabase
def insert_tweets_to_supabase(tweets):
    # Create records to insert
    records = []
    for tweet in tweets:
        tweet_id = tweet.get("id", "")
        url = tweet.get("url", "")
        content = tweet.get("text", "")
        
        # Skip empty content
        if not content.strip():
            continue
            
        # Create embeddings for the content
        try:
            embedding_response = embeddings.embed_query(content)
            
            # Create record with all required fields
            record = {
                "content": content,
                "tweet_id": tweet_id,
                "url": url,
                "embedding": embedding_response,
                "metadata": {
                    "id": tweet_id,
                    "url": url,
                    "authorUsername": tweet.get("authorUsername", ""),
                    "authorFullname": tweet.get("authorFullname", ""),
                    "createdAt": tweet.get("createdAt", ""),
                    "source": "twitter"
                }
            }
            records.append(record)
        except Exception as e:
            print(f"❌ Error creating embedding for tweet {tweet_id}: {str(e)}")
            continue
    
    if not records:
        return 0
        
    # Insert records in batches
    batch_size = 10
    successful_inserts = 0
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            response = supabase.table("tweets").insert(batch).execute()
            successful_inserts += len(response.data)
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        except Exception as e:
            print(f"\n❌ Error inserting batch {i//batch_size + 1}: {str(e)}")
            continue
    
    return successful_inserts

# Function to convert Twitter data to LangChain documents
def convert_tweets_to_documents(tweets):
    documents = []
    for tweet in tweets:
        # Create a document with the tweet content as page_content
        # and other tweet data as metadata
        doc = Document(
            page_content=tweet.get("text", ""),
            metadata={
                "id": tweet.get("id", ""),
                "url": tweet.get("url", ""),
                "authorUsername": tweet.get("authorUsername", ""),
                "authorFullname": tweet.get("authorFullname", ""),
                "createdAt": tweet.get("createdAt", ""),
                "source": "twitter"
            }
        )
        documents.append(doc)
    return documents

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("🔄 Loading Twitter data from API...")
# Fetch tweets from API
tweets = fetch_tweet_data()

if not tweets:
    print("❌ No tweets found. Please check the API response.")
    exit(1)

# Filter out tweets that already exist in the database
print("🔄 Checking for existing tweets...")
new_tweets = []
for tweet in tqdm(tweets, desc="Checking duplicates"):
    if not check_tweet_exists(tweet.get("id")):
        new_tweets.append(tweet)
    
print(f"🔍 Found {len(new_tweets)} new tweets out of {len(tweets)} total")

if not new_tweets:
    print("✅ No new tweets to add. Exiting.")
    exit(0)

print("\n🔍 Inserting tweets directly into Supabase...")
inserted_count = insert_tweets_to_supabase(new_tweets)
print(f"\n✅ Twitter data ingestion complete! Added {inserted_count} new tweets to Supabase.")

# The code below is commented out as we're now using direct insertion instead of SupabaseVectorStore
"""
# Convert tweets to LangChain documents
print("🔄 Converting tweets to documents...")
documents = convert_tweets_to_documents(new_tweets)
print(f"📚 Converted {len(documents)} tweets to documents")

# Split content if needed (many tweets are short, so this might not be necessary)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
print(f"✂️ Split into {len(docs)} chunks")

# Clean the documents to remove problematic characters
print("🧹 Cleaning document text to remove problematic characters...")
docs = clean_documents(docs)

# Create or check if the table exists
print("🔄 Checking if 'tweets' table exists in Supabase...")
try:
    # You may need to adjust this depending on your Supabase setup
    response = supabase.table("tweets").select("count", count="exact").limit(1).execute()
    print("✅ Table 'tweets' exists")
except Exception as e:
    print(f"ℹ️ Note: {str(e)}")
    print("🔄 Creating 'tweets' table may be necessary in Supabase")

print("\n🔍 Processing and uploading tweets to Supabase...")
# Process chunks in batches to show progress
batch_size = 10
total_batches = (len(docs) + batch_size - 1) // batch_size

for i in tqdm(range(0, len(docs), batch_size), desc="Uploading batches", total=total_batches):
    batch = docs[i:i+batch_size]
    try:
        # store chunks in vector store
        vector_store = SupabaseVectorStore.from_documents(
            batch,
            embeddings,
            client=supabase,
            table_name="tweets",  # Using a different table for tweets
            query_name="match_tweets",  # New query name for tweets
            chunk_size=1000,
        )
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    except Exception as e:
        print(f"\n❌ Error processing batch {i//batch_size + 1}: {str(e)}")
        continue
"""