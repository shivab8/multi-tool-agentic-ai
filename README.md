# Multi-Tool AI Agent

A state-of-the-art Agentic AI application built with **LangGraph**, **Streamlit**, and the **Model Context Protocol (MCP)**. This intelligent chatbot orchestrates multiple tools, persists conversation state, and seamlessly connects to external services.

## Features
- **Agentic Workflow (LangGraph)**: Stateful conversational agent capable of tool calling and complex multi-step reasoning.
- **Real-Time Streaming UI**: Built with Streamlit, featuring asynchronous background loops for smooth token streaming.
- **Model Context Protocol (MCP)**: Native integration with Notion via `@notionhq/notion-mcp-server` for seamless workspace interaction.
- **RAG Integration**: Local FAISS vector store integration (`search_knowledge_base`) for retrieving custom knowledge.
- **Live APIs**: Integrated tool to fetch real-time stock prices (`get_stock_price`) via Alpha Vantage.
- **Long-Term Memory**: Conversation checkpointing utilizing an asynchronous `sqlite` database.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/multi-tool-ai-agent.git
   cd multi-tool-ai-agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory (you can use `.env.example` as a template) and add your API keys:
   ```ini
   OPENAI_API_KEY=your_openai_api_key
   ALPHA_VANTAGE_API_KEY=your_alphavantage_api_key
   NOTION_API_KEY=your_notion_integration_token
   ```

4. **MCP Requirements:**
   The Notion MCP server requires Node.js `npx` to be installed on your machine.
   Ensure you have [Node.js installed](https://nodejs.org/).

## Running the App
Start the Streamlit development server:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.
