# utils.py

import re
from langchain.schema import Document

def clean_text(text):
    """Clean text of problematic characters"""
    if text is None:
        return ""
    # Remove null bytes and other problematic control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # Replace other potentially problematic Unicode characters
    text = text.replace('\\u0000', '')
    return text

def clean_documents(docs):
    """Clean document content and metadata"""
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        # Clean metadata values that are strings
        for key, value in doc.metadata.items():
            if isinstance(value, str):
                doc.metadata[key] = clean_text(value)
    return docs

def convert_tweets_to_documents(tweets):
    """Convert Twitter data to LangChain documents"""
    documents = []
    for tweet in tweets:
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

def get_all_tweets(supabase, limit: int = 50):
    """Retrieve tweets from database"""
    try:
        resp = supabase.table("tweets").select("*").limit(limit).execute()
        return resp.data or []
    except Exception as e:
        print("Error fetching tweets:", e)
        return []