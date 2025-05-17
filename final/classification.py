# import logging
# from typing import List, Dict
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )
# logger = logging.getLogger("sui-api-monitor")

# # Initialize LLM (GPT-4o)
# llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# # Define features
# FEATURES = [
#     {
#         "title": "Check Wallet portfolio",
#         "path": "/balance/:address",
#         "description": "Check the balance of any Sui wallet address for SUI or other tokens",
#         "keys": ["balance", "wallet", "address", "check", "check balance", "sui balance", "token", "portfolio"],
#         "icon": "💰"
#     },
#     {
#         "title": "Check my portfolio",
#         "path": "/balance/:myAddress",
#         "description": "Check the balance of your own Sui wallet",
#         "keys": ["my balance", "my wallet", "check my", "my sui", "my tokens", "my portfolio"],
#         "icon": "💰"
#     },
#     {
#         "title": "Check Package",
#         "path": "/package/:packageId",
#         "description": "Inspect details of a Sui Move package (e.g., smart contract)",
#         "keys": ["package", "package id", "check package", "smart contract", "move package", "contract details"],
#         "icon": "📦"
#     },
#     {
#         "title": "My Packages",
#         "path": "/packages/:myAddress",
#         "description": "List all Sui Move packages published by your wallet",
#         "keys": ["my packages", "my contracts", "published packages", "my move", "my smart contracts"],
#         "icon": "📦"
#     },
#     {
#         "title": "Check Packages on a wallet",
#         "path": "/packages/:walletAddress",
#         "description": "List all Sui Move packages for a given wallet address",
#         "keys": ["packages", "check packages", "wallet packages", "address packages"],
#         "icon": "📦"
#     },
#     {
#         "title": "Swap Tokens",
#         "path": "/swap/:address",
#         "description": "Swap tokens on a Sui-based DEX or AMM",
#         "keys": ["swap", "trade", "exchange", "swap tokens", "dex", "amm", "token swap"],
#         "icon": "🔄"
#     },
#     {
#         "title": "Send Tokens",
#         "path": "/send/:address",
#         "description": "Send tokens from your wallet to another Sui address",
#         "keys": ["send", "transfer", "send tokens", "send sui", "transfer tokens"],
#         "icon": "🚀"
#     },
#     {
#         "title": "Transaction History",
#         "path": "/transactions/:address",
#         "description": "View transaction history for a Sui wallet address",
#         "keys": ["transactions", "history", "transaction history", "tx history", "wallet history"],
#         "icon": "📜"
#     },
#     {
#         "title": "Learn Sui Move",
#         "path": "/learn/move",
#         "description": "Get educational content about Sui Move programming",
#         "keys": ["learn", "sui move", "move programming", "learn move", "coding", "tutorial"],
#         "icon": "📚"
#     },
#     {
#         "title": "Explore Sui Network",
#         "path": "/explore",
#         "description": "Discover stats and updates about the Sui blockchain",
#         "keys": ["explore", "sui network", "network stats", "sui stats", "blockchain info"],
#         "icon": "🌐"
#     }
# ]

# # Prompt template
# PROMPT_TEMPLATE = """You are Fly Explorer, a Web3 routing assistant for the Sui blockchain, with a Gen Z, Viet–English crypto vibe. Your job is to match the user’s query to the best feature based on keywords and semantic meaning. The query might mix Vietnamese and English, use crypto slang (e.g., HODL, LFG), or be super casual. Focus on Sui-related tasks like checking balances, Move packages, token swaps, or learning.

# User query: "{user_query}"

# Features:
# {features}

# Return only the feature title that best matches the user query. If no feature matches, return "None".
# """

# # ---------------------- FastAPI & CORS ----------------------
# app = FastAPI(title="Sui Network AI Assistant API")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # production: đổi thành domain frontend của bạn
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ---------------------- Models ----------------------
# class QueryRequest(BaseModel):
#     input: str

# class QueryResponse(BaseModel):
#     keys: List[str]

# # ---------------------- API Endpoints ----------------------
# @app.post("/api/query", response_model=QueryResponse)
# async def query_feature(req: QueryRequest):
#     """
#     POST /api/query
#     Body JSON: { "input": "Your question here" }
#     Returns the keys of the best-matching feature.
#     """
#     try:
#         logger.info(f"Processing query: {req.input}")
        
#         # Format features for prompt
#         features_str = "\n".join(
#             f"{i+1}. Title: {f['title']}\n"
#             f"   Path: {f['path']}\n"
#             f"   Description: {f['description']}\n"
#             f"   Keys: {f['keys']}\n"
#             f"   Icon: {f['icon']}"
#             for i, f in enumerate(FEATURES)
#         )
        
#         # Create prompt
#         prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#         formatted_prompt = prompt.format(
#             user_query=req.input,
#             features=features_str
#         )
        
#         # Call GPT-4o
#         result = await llm.ainvoke(formatted_prompt)
#         feature_title = result.content.strip()
        
#         # Find the feature to get its keys
#         selected_feature = next((f for f in FEATURES if f["title"] == feature_title), None)
#         keys = selected_feature["keys"] if selected_feature else []
        
#         logger.info(f"Selected feature: {feature_title}, Keys: {keys}")
#         return QueryResponse(keys=keys)
    
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return QueryResponse(keys=[])

# # ---------------------- Run ----------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("classification:app", host="0.0.0.0", port=8000, reload=True)

import logging
from typing import List, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sui-api-monitor")

# Initialize LLM (GPT-4o)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Define features
FEATURES = [
    {
        "title": "Check Wallet portfolio",
        "path": "/balance/:address",
        "description": "Check the balance of any Sui wallet address for SUI or other tokens",
        "keys": ["balance", "wallet", "address", "check", "check balance", "sui balance", "token", "portfolio"],
        "icon": "💰"
    },
    {
        "title": "Check my portfolio",
        "path": "/balance/:myAddress",
        "description": "Check the balance of your own Sui wallet",
        "keys": ["my balance", "my wallet", "check my", "my sui", "my tokens", "my portfolio"],
        "icon": "💰"
    },
    {
        "title": "Check Package",
        "path": "/package/:packageId",
        "description": "Inspect details of a Sui Move package (e.g., smart contract)",
        "keys": ["package", "package id", "check package", "smart contract", "move package", "contract details"],
        "icon": "📦"
    },
    {
        "title": "My Packages",
        "path": "/packages/:myAddress",
        "description": "List all Sui Move packages published by your wallet",
        "keys": ["my packages", "my contracts", "published packages", "my move", "my smart contracts"],
        "icon": "📦"
    },
    {
        "title": "Check Packages on a wallet",
        "path": "/packages/:walletAddress",
        "description": "List all Sui Move packages for a given wallet address",
        "keys": ["packages", "check packages", "wallet packages", "address packages"],
        "icon": "📦"
    }
]

# Prompt template
PROMPT_TEMPLATE = """You are Fly Explorer, a Web3 routing assistant for the Sui blockchain, with a Gen Z, Viet–English crypto vibe. Your job is to match the user’s query to the best feature based on keywords and semantic meaning. The query might mix Vietnamese and English, use crypto slang (e.g., HODL, LFG), or be super casual. Focus on Sui-related tasks like checking balances, Move packages, token swaps, or learning.

User query: "{user_query}"

Features:
{features}

Return only the feature title that best matches the user query. If no feature matches, return "None".
"""

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
    key: str

# ---------------------- API Endpoints ----------------------
@app.post("/api/query", response_model=QueryResponse)
async def query_feature(req: QueryRequest):
    """
    POST /api/query
    Body JSON: { "input": "Your question here" }
    Returns a single key from the best-matching feature.
    """
    try:
        logger.info(f"Processing query: {req.input}")
        
        # Format features for prompt
        features_str = "\n".join(
            f"{i+1}. Title: {f['title']}\n"
            f"   Path: {f['path']}\n"
            f"   Description: {f['description']}\n"
            f"   Keys: {f['keys']}\n"
            f"   Icon: {f['icon']}"
            for i, f in enumerate(FEATURES)
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        formatted_prompt = prompt.format(
            user_query=req.input,
            features=features_str
        )
        
        # Call GPT-4o
        result = await llm.ainvoke(formatted_prompt)
        feature_title = result.content.strip()
        
        # Find the feature to get its first key
        selected_feature = next((f for f in FEATURES if f["title"] == feature_title), None)
        key = selected_feature["keys"][0] if selected_feature and selected_feature["keys"] else ""
        
        logger.info(f"Selected feature: {feature_title}, Key: {key}")
        return QueryResponse(key=key)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(key="")

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("classification:app", host="0.0.0.0", port=8000, reload=True)