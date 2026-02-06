# 🤖 Autonomous Chat-with-PDF Agent

An intelligent RAG (Retrieval-Augmented Generation) agent built with **LangGraph** and **LangChain** that can maintain long-term memory and provide context-aware answers from Node.js documentation.

## 🚀 Features
- **Stateful Management**: Uses LangGraph's `StateGraph` for managing conversation flow.
- **LTM (Long-Term Memory)**: Integrated with MongoDB for persistent chat history.
- **Hybrid RAG**: Combines vector search (FAISS) with LLM reasoning.
- **Docker Ready**: Pre-configured Dockerfile for seamless deployment.

## 🛠️ Tech Stack
- **AI Framework**: LangGraph, LangChain
- **LLM**: OpenAI GPT-4o-mini
- **Database**: MongoDB (Memory), FAISS (Vector Store)
- **Backend**: FastAPI, Python 3.11

## 🏃 How to Run
1. Clone the repo: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your `.env` variables (OPENAI_API_KEY, MONGODB_URI).
4. Run ingestion: `python src/ingest.py`
5. Start server: `uvicorn src.server:app --reload`
