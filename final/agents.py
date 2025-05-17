import logging
from typing import Optional
import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sui-api-monitor")

# ---------------------- LLM & System Prompt ----------------------
# Get API key from environment variable or set it directly here
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    openai_api_key = input("Please enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = openai_api_key

llm = ChatOpenAI(temperature=0.2)

system_prompt = """Bạn là một trợ lý AI chuyên nghiệp, đồng hành cùng một Web3 educator & content creator tên Fly Explorer. Phong cách của bạn phải pha trộn giữa:

    • Ngôn ngữ Gen Z, flex giữa Việt–Anh, nhất là crypto terms (HODL, DeFi, LFG 🚀).
    • Tone: vui vẻ, khuyến khích, chill như đang chat với homie.
    • Văn phong: không gò bó, chèn icon, dễ share lên Facebook/X/Substack.
    • Ưu tiên storytelling, dẫn dắt có lớp lang, gợi ý meme/diagram nếu hợp.
    • Chủ đề: Sui Move, blockchain dev, AI+Web3, coding edu, onchain game, startup mindset, hệ sinh thái Việt Nam.

Mục tiêu: Giúp Fly Explorer truyền tải insight Web3 sâu sắc mà vẫn relatable, như đang kể chuyện cho fan crypto. Tập trung vào các tác vụ blockchain như check balance, swap token, send token trên Sui network. Nếu user hỏi ngoài scope, trả lời ngắn gọn, vui vẻ, và gợi ý thử lại với các lệnh blockchain!

Ví dụ:
- User: "Check bao nhiêu SUI trong ví 0x123 đi bro"
- Output: "🔥 Đang check ví 0x123... Boom! Có 100.5 SUI nè bro! HODL tight nhé! 🤑"
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
    description="Check the balance of a Sui wallet address for SUI tokens or other tokens."
)
def check_balance(wallet_address: str, token_contract: Optional[str] = None):
    """Check the balance of a Sui wallet address.
    
    Args:
        wallet_address: The Sui wallet address to query (e.g., '0x123...').
        token_contract: Optional token contract address for non-SUI tokens (e.g., USDC).
    
    Returns:
        A string with balance details and a list of balance data.
    """
    try:
        logger.info(f"Checking balance for wallet: {wallet_address}, token: {token_contract or 'SUI'}")
        # Placeholder: Replace with Sui SDK/RPC call to query balance
        print(f"Checking balance for {wallet_address}, token: {token_contract or 'SUI'}")
        balance = {"wallet": wallet_address, "token": token_contract or "SUI", "amount": "100.5"}  # Mock data
        serialized = f"Wallet: {wallet_address}\nToken: {balance['token']}\nBalance: {balance['amount']}"
        logger.info(f"Balance retrieved: {serialized[:100]}...")
        return serialized, [balance]
    except Exception as e:
        logger.error(f"Error in check_balance: {str(e)}")
        error_message = f"Oops, couldn't check balance for {wallet_address}! Error: {str(e)} 😅 Try again or check the address!"
        return error_message, []

@tool(
    response_format="content_and_artifact",
    description="Swap tokens on a Sui-based DEX or AMM."
)
def swap(wallet_address: str, input_token: str, output_token: str, amount: float, slippage: float = 0.5):
    """Swap tokens on a Sui-based decentralized exchange.
    
    Args:
        wallet_address: The Sui wallet address performing the swap.
        input_token: Contract address of the input token (e.g., SUI).
        output_token: Contract address of the output token (e.g., USDC).
        amount: Amount of input token to swap.
        slippage: Maximum slippage percentage (default: 0.5%).
    
    Returns:
        A string with swap details and a list of swap data.
    """
    try:
        logger.info(f"Swapping {amount} {input_token} to {output_token} for {wallet_address}")
        # Placeholder: Replace with Sui DEX/AMM contract call
        print(f"Swapping {amount} {input_token} to {output_token} for {wallet_address}, slippage: {slippage}%")
        swap_data = {
            "wallet": wallet_address,
            "input_token": input_token,
            "output_token": output_token,
            "input_amount": amount,
            "output_amount": amount * 0.995,  # Mock data
            "slippage": slippage
        }
        serialized = (
            f"Swap for {wallet_address}\n"
            f"From: {amount} {input_token}\n"
            f"To: {swap_data['output_amount']} {output_token}\n"
            f"Slippage: {slippage}%"
        )
        logger.info(f"Swap executed: {serialized[:100]}...")
        return serialized, [swap_data]
    except Exception as e:
        logger.error(f"Error in swap: {str(e)}")
        error_message = f"Swap failed for {wallet_address}! Error: {str(e)} 🚫 Maybe check your balance or slippage?"
        return error_message, []

@tool(
    response_format="content_and_artifact",
    description="Send tokens from one Sui wallet to another."
)
def send_token(sender_address: str, recipient_address: str, token_contract: str, amount: float):
    """Send tokens from one Sui wallet to another.
    
    Args:
        sender_address: The Sui wallet address sending the tokens.
        recipient_address: The Sui wallet address receiving the tokens.
        token_contract: Contract address of the token to send (e.g., SUI).
        amount: Amount of tokens to send.
    
    Returns:
        A string with transaction details and a list of transaction data.
    """
    try:
        logger.info(f"Sending {amount} {token_contract} from {sender_address} to {recipient_address}")
        # Placeholder: Replace with Sui transaction call
        print(f"Sending {amount} {token_contract} from {sender_address} to {recipient_address}")
        tx_data = {
            "sender": sender_address,
            "recipient": recipient_address,
            "token": token_contract,
            "amount": amount,
            "tx_hash": "0xabc123..."  # Mock data
        }
        serialized = (
            f"Transaction from {sender_address}\n"
            f"To: {recipient_address}\n"
            f"Token: {token_contract}\n"
            f"Amount: {amount}\n"
            f"Tx Hash: {tx_data['tx_hash']}"
        )
        logger.info(f"Transaction sent: {serialized[:100]}...")
        return serialized, [tx_data]
    except Exception as e:
        logger.error(f"Error in send_token: {str(e)}")
        error_message = f"Couldn't send {amount} {token_contract} from {sender_address} to {recipient_address}! Error: {str(e)} 😬 Check your balance or addresses!"
        return error_message, []

# ---------------------- Agent ----------------------
tools = [check_balance, swap, send_token]
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

# ---------------------- Run ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents:app", host="0.0.0.0", port=8000, reload=True)


{
    "features": [{
        "title": "Check Wallet portfolio",
        "path": "/balance/:address",
        "description": "Check the balance of a given address",
        "keys": ["balance", "wallet", "address", "balances", "check", "check balance", "check wallet balance", "token"],
        "icon": "💰"
    },{
        "title": "Check my portfolio",
        "path": "/balance/:myAddress",
        "description": "Check the balance of your wallet",
        "keys": ["balance", "wallet", "address", "balances", "check", "check balance", "check wallet balance", "token"],
        "icon": "💰"
    },{
        "title": "Package",
        "path": "/package/:packageId",
        "description": "Check the details of a given package",
        "keys": ["package", "package id", "check", "check package", "check package id", "call"],
        "icon": "📦"
    },
    {
        "title": "My Packages",
        "path": "/many/:myAddress",
        "description": "Check the details of all packages",
        "keys": ["my packages", "packages", "check", "check packages", "call"],
        "icon": "📦"
    },{
        "title": "Check Packages on a wallet address",
        "path": "/many/:walletAddress",
        "description": "Check the details of all packages on a given wallet address",
        "keys": ["packages", "check", "check packages", "call"],
        "icon": "📦"
    }
]

}