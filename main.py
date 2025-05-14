# main.py

import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase.client import create_client
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

# ---------------------- Load config ----------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

print("➜ SUPABASE_URL:", SUPABASE_URL)
print("➜ SUPABASE_KEY loaded?:", bool(SUPABASE_KEY))

# ---------------------- Supabase & Embeddings ----------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
"""

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ---------------------- Helper lấy raw tweets ----------------------
def get_all_tweets(limit: int = 50):
    try:
        resp = supabase.table("tweets").select("*").limit(limit).execute()
        return resp.data or []
    except Exception as e:
        print("Error fetching tweets:", e)
        return []

# ---------------------- Định nghĩa tools có description ----------------------
@tool(
    response_format="content_and_artifact",
    description="Retrieve information from documents related to a query from Supabase vector store."
)
def retrieve_documents(query: str):
    docs = document_vector_store.similarity_search(query, k=5)
    serialized = "\n\n".join(
        f"Source: {d.metadata}\nContent: {d.page_content}"
        for d in docs
    )
    return serialized, docs

@tool(
    response_format="content_and_artifact",
    description="Retrieve tweets semantically similar to the query from Supabase vector store."
)
def retrieve_tweets(query: str):
    tweets = tweets_vector_store.similarity_search(query, k=10)
    serialized = "\n\n".join(
        f"Tweet ID: {t.metadata.get('id','?')}\nAuthor: {t.metadata.get('authorFullname','')} (@{t.metadata.get('authorUsername','')})\nDate: {t.metadata.get('createdAt','')}\nURL: {t.metadata.get('url','?')}\n{t.page_content}"
        for t in tweets
    )
    return serialized, tweets

@tool(
    response_format="content_and_artifact",
    description="Get a summary of raw tweets by category (all, basecamp, defi, community, announcements)."
)
def get_tweet_summary(category: str = "all"):
    raw = get_all_tweets(50)
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

# ---------------------- Tạo agent ----------------------
tools = [retrieve_documents, retrieve_tweets, get_tweet_summary]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------------------- FastAPI & CORS ----------------------
app = FastAPI(title="Sui Network AI Assistant API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # production: đổi thành domain frontend của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- Models ----------------------
class QueryRequest(BaseModel):
    input: str

class QueryResponse(BaseModel):
    output: str

# ---------------------- Endpoints ----------------------
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

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

