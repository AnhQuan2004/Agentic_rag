# ingestion.py

import json
import time
import requests
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from config import get_supabase_client, get_embeddings, TWEETS_API_URL
from utils import clean_text, clean_documents, convert_tweets_to_documents

# Initialize clients from shared config
supabase = get_supabase_client()
embeddings = get_embeddings()

def fetch_tweet_data(address: str = "0x000123456789") -> List[Dict]:
    """Fetch tweet data from external API"""
    params = {"address": address}
    
    try:
        response = requests.get(TWEETS_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"📊 Loaded {len(data['data'])} tweets from API")
        return data['data']
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data from API: {str(e)}")
        return []
    except (KeyError, json.JSONDecodeError) as e:
        print(f"❌ Error parsing API response: {str(e)}")
        return []

def check_tweet_exists(tweet_id: str) -> bool:
    """Check if a tweet already exists in the database"""
    try:
        response = supabase.table("tweets").select("id").eq("tweet_id", str(tweet_id)).execute()
        return len(response.data) > 0
    except Exception as e:
        print(f"❌ Error checking for existing tweet: {str(e)}")
        return False

def insert_tweets_to_supabase(tweets: List[Dict]) -> int:
    """Insert tweets directly into Supabase with embeddings"""
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

def ingest_tweets(address: Optional[str] = None) -> Tuple[int, int]:
    """Main function to ingest tweets from API to database
    
    Returns:
        Tuple containing (total_tweets, inserted_tweets)
    """
    # Fetch tweets from API
    print("🔄 Loading Twitter data from API...")
    tweets = fetch_tweet_data(address) if address else fetch_tweet_data()

    if not tweets:
        print("❌ No tweets found. Please check the API response.")
        return 0, 0

    # Filter out tweets that already exist in the database
    print("🔄 Checking for existing tweets...")
    new_tweets = []
    for tweet in tqdm(tweets, desc="Checking duplicates"):
        if not check_tweet_exists(tweet.get("id")):
            new_tweets.append(tweet)
        
    print(f"🔍 Found {len(new_tweets)} new tweets out of {len(tweets)} total")

    if not new_tweets:
        print("✅ No new tweets to add.")
        return len(tweets), 0

    print("\n🔍 Inserting tweets directly into Supabase...")
    inserted_count = insert_tweets_to_supabase(new_tweets)
    print(f"\n✅ Twitter data ingestion complete! Added {inserted_count} new tweets to Supabase.")
    
    return len(tweets), inserted_count

# For CLI usage
if __name__ == "__main__":
    total, inserted = ingest_tweets()
    print(f"Processed {total} tweets, inserted {inserted} new tweets")