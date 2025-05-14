# main.py

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

# Import shared components
from config import get_supabase_client, get_embeddings, print_config
from utils import get_all_tweets
from ingestion import ingest_tweets, fetch_tweet_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sui-api-monitor")

# ---------------------- Initialize config ----------------------
print_config()
supabase = get_supabase_client()
embeddings = get_embeddings()

# ---------------------- Vector stores ----------------------
document_vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="tweets",
    query_name="match_tweets",
)
tweets_vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="tweets",
    query_name="match_tweets",
)

# ---------------------- LLM & System Prompt ----------------------
llm = ChatOpenAI(temperature=0.2)

system_prompt = """Bạn là một trợ lý AI chuyên nghiệp, đồng hành cùng một Web3 educator & content creator tên Fly explorer. Phong cách của bạn phải pha trộn giữa:

    • ngôn ngữ Gen Z, flex giữa Việt–Anh, nhất là crypto terms.
    • Tone: vui vẻ, khuyến khích, thoải mái.
    • Văn phong: không gò bó, chèn icon, dễ share Facebook/Twitter/Substack.
    • Ưu tiên storytelling, dẫn dắt, có lớp lang, gợi ý meme/diagram nếu hợp.
    • Chủ đề: Sui Move, blockchain dev, AI+Web3, coding edu, onchain game, startup mindset, hệ sinh thái Việt Nam…

Mục tiêu: giúp Harry truyền insight Web3 sâu sắc mà vẫn relatable.

Quan trọng: Khi semantic search không tìm thấy kết quả, hãy sử dụng công cụ get_recent_tweets để xem các bài đăng mới nhất thay vì báo lỗi. Luôn cố gắng cung cấp thông tin hữu ích cho người dùng.
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------------------- Tools ----------------------
@tool(
    response_format="content_and_artifact",
    description="Retrieve information from documents related to a query from Supabase vector store."
)
def retrieve_documents(query: str):
    try:
        docs = document_vector_store.similarity_search(query, k=5)
        
        # Log retrieval results for debugging
        logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
        
        if not docs:
            # No documents found
            logger.warning(f"No documents found for query: '{query}'")
            fallback_message = f"No documents found matching '{query}'. The database may not contain relevant information on this topic."
            return fallback_message, []
            
        serialized = "\n\n".join(
            f"Source: {d.metadata}\nContent: {d.page_content}"
            for d in docs
        )
        
        # Log a sample of what was found
        logger.info(f"First document sample: {docs[0].page_content[:100]}...")
        
        return serialized, docs
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error in retrieve_documents: {str(e)}")
        error_message = f"Error retrieving documents for '{query}': {str(e)}"
        return error_message, []

@tool(
    response_format="content_and_artifact",
    description="Retrieve tweets semantically similar to the query from Supabase vector store."
)
def retrieve_tweets(query: str):
    try:
        tweets = tweets_vector_store.similarity_search(query, k=10)
        
        # Log retrieval results for debugging
        logger.info(f"Retrieved {len(tweets)} tweets for query: '{query}'")
        
        if not tweets:
            # No tweets found
            logger.warning(f"No tweets found for query: '{query}'")
            fallback_message = f"No tweets found matching '{query}'. Try a different search term or check if tweets have been ingested."
            return fallback_message, []
            
        serialized = "\n\n".join(
            f"Tweet ID: {t.metadata.get('id','?')}\nAuthor: {t.metadata.get('authorFullname','')} (@{t.metadata.get('authorUsername','')})\nDate: {t.metadata.get('createdAt','')}\nURL: {t.metadata.get('url','?')}\n{t.page_content}"
            for t in tweets
        )
        
        # Log a sample of what was found
        if tweets:
            logger.info(f"First tweet sample: {tweets[0].page_content[:100]}...")
        
        return serialized, tweets
        
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error in retrieve_tweets: {str(e)}")
        error_message = f"Error retrieving tweets for '{query}': {str(e)}"
        return error_message, []

@tool(
    response_format="content_and_artifact",
    description="Get a summary of raw tweets by category (all, basecamp, defi, community, announcements)."
)
def get_tweet_summary(category: str = "all"):
    raw = get_all_tweets(supabase, 50)
    if category.lower() == "all":
        filtered = raw
    else:
        keywords = {
            "basecamp": ["basecamp","camp","dubai","event"],
            "defi": ["defi","finance","swap","lending","trading","scallop"],
            "community": ["community","suifam","vietnam","japan","meetup"],
            "announcements": ["live","launched","announcement","new","update"],
        }.get(category.lower(), [category.lower()])
        filtered = [t for t in raw if any(kw in t.get("page_content","").lower() for kw in keywords)]
    previews = [
        f"Author: {t.get('metadata', {}).get('authorFullname','')} (@{t.get('metadata', {}).get('authorUsername','')})\nDate: {t.get('metadata', {}).get('createdAt','')}\nURL: {t.get('metadata', {}).get('url','?')}\n{t.get('page_content','')}"
        for t in filtered
    ]
    header = (
        f"Found {len(filtered)} tweets for '{category}'"
        if category.lower() != "all"
        else f"Found {len(filtered)} tweets"
    )
    return header + ":\n\n" + "\n\n".join(previews), filtered

@tool(
    response_format="content_and_artifact", 
    description="Get the most recent tweets from the database without semantic search."
)
def get_recent_tweets(limit: int = 10):
    """Get the most recent tweets from the database directly, without semantic search.
    Useful when semantic search doesn't return results or you want to see the latest content."""
    try:
        # Direct database query to get the most recent tweets
        response = supabase.table("tweets").select("*").order("created_at", desc=True).limit(limit).execute()
        tweets = response.data
        
        if not tweets or len(tweets) == 0:
            return "No tweets found in the database. Try ingesting tweets first.", []
            
        logger.info(f"Retrieved {len(tweets)} recent tweets directly from database")
        
        # Format tweets for display
        formatted_tweets = []
        for tweet in tweets:
            metadata = tweet.get('metadata', {})
            content = tweet.get('content', '')
            if not content and 'page_content' in tweet:
                content = tweet.get('page_content', '')
                
            formatted = {
                'page_content': content,
                'metadata': metadata
            }
            formatted_tweets.append(formatted)
            
        # Serialize for display
        serialized = "\n\n".join(
            f"Tweet ID: {t.get('metadata', {}).get('id','?')}\n"
            f"Author: {t.get('metadata', {}).get('authorFullname','')} (@{t.get('metadata', {}).get('authorUsername','')})\n"
            f"Date: {t.get('metadata', {}).get('createdAt','')}\n"
            f"URL: {t.get('metadata', {}).get('url','?')}\n"
            f"{t.get('page_content', '')}"
            for t in formatted_tweets
        )
        
        return serialized, formatted_tweets
        
    except Exception as e:
        logger.error(f"Error in get_recent_tweets: {str(e)}")
        return f"Error retrieving recent tweets: {str(e)}", []

# ---------------------- Agent ----------------------
tools = [retrieve_documents, retrieve_tweets, get_tweet_summary, get_recent_tweets]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------- FastAPI & CORS ----------------------
app = FastAPI(title="Sui Network AI Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production: đổi thành domain frontend của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Models ----------------------
class QueryRequest(BaseModel):
    input: str

class QueryResponse(BaseModel):
    output: str

class IngestTweetsRequest(BaseModel):
    address: Optional[str] = None

class IngestTweetsResponse(BaseModel):
    total_tweets: int
    new_tweets: int
    status: str

# ---------------------- API Endpoints ----------------------
@app.post("/api/query", response_model=QueryResponse)
def query_agent(req: QueryRequest):
    """
    POST /api/query
    Body JSON: { "input": "Your question here" }
    """
    # Invoke agent, luôn bắt đầu với chat_history=[]
    result = agent_executor.invoke({
        "input": req.input,
        "chat_history": []
    })
    return QueryResponse(output=result["output"])

# ---------------------- New Ingestion Endpoints ----------------------
@app.post("/api/ingest/tweets", response_model=IngestTweetsResponse)
async def ingest_tweets_endpoint(
    req: IngestTweetsRequest, 
    background_tasks: BackgroundTasks
):
    """
    POST /api/ingest/tweets
    Optional Body: { "address": "0x123..." }
    Fetches tweets from the API and ingests them into Supabase
    """
    # Run in background to avoid timeout
    background_tasks.add_task(ingest_tweets, req.address)
    
    return IngestTweetsResponse(
        total_tweets=0,  # These will be determined by the background task
        new_tweets=0,
        status="Ingestion started in background. Check logs for progress."
    )

@app.post("/api/ingest/tweets/sync", response_model=IngestTweetsResponse)
def ingest_tweets_sync(req: IngestTweetsRequest):
    """
    POST /api/ingest/tweets/sync
    Optional Body: { "address": "0x123..." }
    Synchronously fetches and ingests tweets (may timeout for large datasets)
    """
    total, inserted = ingest_tweets(req.address)
    
    return IngestTweetsResponse(
        total_tweets=total,
        new_tweets=inserted,
        status="complete" if total > 0 else "no_tweets_found"
    )

# ---------------------- Debug Endpoints ----------------------
@app.get("/debug/retrieve_tweets")
def debug_tweets(q: str):
    """
    Debug endpoint: xem vector-store trả về bao nhiêu tweet cho query=?
    GET /debug/retrieve_tweets?q=Sui
    """
    serialized, docs = retrieve_tweets(q)
    return {
        "query": q,
        "count": len(docs),
        "preview": serialized[:500]
    }

@app.get("/debug/retrieve_documents")
def debug_docs(q: str):
    """
    Debug endpoint: xem vector-store documents cho query=?
    GET /debug/retrieve_documents?q=move
    """
    serialized, docs = retrieve_documents(q)
    return {
        "query": q,
        "count": len(docs),
        "preview": serialized[:500]
    }

@app.get("/debug/count_tweets")
def count_tweets():
    """
    Debug endpoint: get count of tweets in database
    GET /debug/count_tweets
    """
    try:
        resp = supabase.table("tweets").select("*", count="exact").limit(1).execute()
        count = resp.count if hasattr(resp, 'count') else 0
        return {"count": count}
    except Exception as e:
        return {"error": str(e)}

# ---------------------- Background Task Functions ----------------------
async def periodic_api_checker():
    """Check the API every 60 seconds for new data"""
    logger.info("Starting periodic API checker")
    last_count = 0
    
    while True:
        try:
            logger.info("Checking API for new data...")
            tweets = fetch_tweet_data()
            current_count = len(tweets)
            
            # Log detailed data about the retrieved tweets
            logger.info(f"API Response Data Summary:")
            logger.info(f"- Total tweets retrieved: {current_count}")
            
            # Log authors distribution
            authors = {}
            for tweet in tweets:
                author = tweet.get('authorUsername', 'unknown')
                authors[author] = authors.get(author, 0) + 1
            logger.info(f"- Author distribution: {authors}")
            
            # Log date range if available
            if tweets and 'createdAt' in tweets[0]:
                dates = sorted([t.get('createdAt', '') for t in tweets if t.get('createdAt')])
                if dates:
                    logger.info(f"- Date range: {dates[0]} to {dates[-1]}")
            
            # Check for new tweets
            if current_count > last_count:
                new_tweets = current_count - last_count
                logger.info(f"🆕 Found {new_tweets} new tweets (total: {current_count})")
                
                # Log a preview of the newest tweets with more details
                if new_tweets > 0:
                    newest = tweets[:min(5, new_tweets)]
                    logger.info("Preview of newest tweets:")
                    for i, tweet in enumerate(newest):
                        logger.info(f"Tweet #{i+1}:")
                        logger.info(f"  ID: {tweet.get('id', '?')}")
                        logger.info(f"  Author: @{tweet.get('authorUsername', '?')} ({tweet.get('authorFullname', '?')})")
                        logger.info(f"  Date: {tweet.get('createdAt', '?')}")
                        logger.info(f"  URL: {tweet.get('url', '?')}")
                        logger.info(f"  Content: {tweet.get('text', '')[:200]}...")
            else:
                logger.info(f"No new tweets found (total: {current_count})")
                
            last_count = current_count
            
        except Exception as e:
            logger.error(f"Error in API checker: {str(e)}")
        
        # Wait for 60 seconds before checking again
        await asyncio.sleep(60)

@app.on_event("startup")
async def start_background_tasks():
    """Start background tasks when the API starts"""
    asyncio.create_task(periodic_api_checker())

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)