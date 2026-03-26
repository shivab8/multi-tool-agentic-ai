from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import aiosqlite
import requests
import asyncio
import threading
import os

load_dotenv()

# Dedicated async loop for backend tasks
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)


# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model = 'gpt-4o')

# -------------------
# 2. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def search_knowledge_base(query: str) -> str:
    """Search internal knowledge base for relevant information."""
    
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local("FAISS_local", embeddings, allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=3)
        results = "\n\n".join([doc.page_content for doc in docs])
        return results if results else "No relevant info found."
    except Exception as e:
        return f"Knowledge base not available. Error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return {"error": "Alpha Vantage API key not configured in environment."}
        
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        r = requests.get(url)
        data = r.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return {"price": float(data["Global Quote"]["05. price"])}
        return {"error": f"Could not fetch price. API response: {data}"}
    except Exception as e:
        return {"error": f"Error fetching price: {str(e)}"}


client = MultiServerMCPClient(
    {
        "notion": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {
                "NOTION_TOKEN": os.getenv("NOTION_API_KEY", "")
            }
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []


mcp_tools = load_mcp_tools()

tools = [search_tool, search_knowledge_base, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------


async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())
