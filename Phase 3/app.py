import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 RAG Chatbot with Local LLM")
st.markdown("Upload documents and ask questions about them using Ollama (LLaMA 3.2).")

# Setup data folder
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize session state variables
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing files..."):
                documents = []
                for uploaded_file in uploaded_files:
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Load document
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding='utf-8')
                        
                    documents.extend(loader.load())
                    os.unlink(tmp_path)
                
                # Split texts
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                # Create embeddings and vector store
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = Chroma.from_documents(splits, embeddings)
                
                st.success(f"Successfully processed {len(uploaded_files)} files!")
        else:
            st.warning("Please upload files first.")

# Chat Interface
st.header("2. Chat with your Data")

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about your documents:"):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if st.session_state.vector_store is None:
            response = "Please upload and process documents first before asking questions."
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    llm = OllamaLLM(model="llama3.2")
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    system_prompt = (
                        "You are an assistant for question-answering tasks. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "If you don't know the answer, say that you don't know. "
                        "Use three sentences maximum and keep the answer concise.\n\n"
                        "{context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    result = rag_chain.invoke({"input": prompt})
                    response = result.get("answer", "No answer generated.")
                except Exception as e:
                    response = f"Error generating response: {str(e)}"
                    
            st.markdown(response)
            
    # Add assistant message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})
