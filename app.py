import streamlit as st
import os
import tempfile
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables (like API keys from .env if present)
load_dotenv()

st.set_page_config(page_title="Robotics PDF RAG System", page_icon="🤖")

st.title("🤖 Robotics PDF Q&A System")
st.write("Upload a PDF document and ask questions about its content. This system uses Retrieval-Augmented Generation (RAG).")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        api_key = None
    
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        
    if not api_key:
        st.error("API Key not found! System Admin needs to configure GOOGLE_API_KEY in Streamlit Secrets or .env file.")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    
    if st.button("Process Document") and uploaded_file and api_key:
        with st.spinner("Processing document..."):
            os.environ["GOOGLE_API_KEY"] = api_key
                
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_filepath = temp_file.name
            
            try:
                # 1. Load the PDF
                st.info("Loading document...")
                loader = PyPDFLoader(temp_filepath)
                documents = loader.load()
                
                # 2. Chunk the text
                st.info("Chunking text...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # 3. Create Embeddings & Vector Store
                st.info("Creating embeddings and vector store...")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Initialize ChromaDB
                vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
                
                # Save retriever in session state
                st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                st.success("Ready! You can now ask questions.")
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                # Clean up temp file
                os.unlink(temp_filepath)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about the document..."):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif "retriever" not in st.session_state:
        st.error("Please upload and process a PDF document first.")
    else:
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3)
                    
                    # Create the RAG chain
                    system_prompt = (
                        "You are an assistant for question-answering tasks, specifically for robotics subjects. "
                        "Always start your answer with 'Hello Faisal Khan,' on the first line, followed by a new line, and then your actual answer. "
                        "Use the following pieces of retrieved context to answer the question. "
                        "If you don't know the answer based on the context, just say that you don't know. "
                        "Use three sentences maximum and keep the answer concise.\n\n"
                        "{context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)

                    rag_chain = (
                        {"context": st.session_state.retriever | format_docs, "input": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    answer = rag_chain.invoke(prompt)
                    
                    st.markdown(answer)
                    # Add assistant response to state
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
