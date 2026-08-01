# Phase 3: RAG Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot using LangChain, Streamlit, ChromaDB, and Local Ollama (`llama3.2`).

## Features
- Upload your own PDF and TXT documents.
- Uses `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) to create a local vector database.
- Uses locally running LLaMA 3.2 via Ollama to answer questions based on the retrieved context.
- Streamlit web interface.

## Quick Start

### 1. Install Dependencies
Make sure you have Ollama installed and `llama3.2` model downloaded (`ollama run llama3.2`). Then install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
streamlit run app.py
```

### 3. Usage
1. Upload your `.pdf` or `.txt` document in the sidebar.
2. Click **Process Documents**.
3. Ask the chatbot questions about the uploaded content!