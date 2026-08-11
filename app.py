import os
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI Configuration
st.set_page_config(page_title="Electrical Engineering AI Assistant", page_icon="⚡")
st.title("⚡ Electrical Engineering AI Assistant")
st.write("Upload your textbook PDF or use the default electrical engineering manual.")

# Sidebar Configuration for Groq API Key
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter Free Groq API Key:", type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

# Main Area File Uploader & URL Option
st.markdown("### 📂 Step 1: Provide your Document")
upload_option = st.radio("Choose how to provide the PDF:", ["Upload PDF File from Computer", "Use Default Textbook URL"])

target_pdf_path = None

if upload_option == "Upload PDF File from Computer":
    uploaded_file = st.file_uploader("Upload your PDF file here", type=["pdf"])
    if uploaded_file is not None:
        local_pdf_path = "user_uploaded_manual.pdf"
        with open(local_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_pdf_path = local_pdf_path
        st.success("File uploaded successfully!")
else:
    # Default PDF URL option
    default_url = "https://mycollegevcampus.com/sjcet/notes/Text_Book_2_Electric_Machinery_And_Power_System_Fundamentals_-_Chapman__S.J..pdf"
    st.info(f"Using default textbook URL: {default_url}")
    local_pdf_path = "downloaded_textbook.pdf"
    try:
        if not os.path.exists(local_pdf_path):
            with st.spinner("Downloading default textbook..."):
                response = requests.get(default_url)
                with open(local_pdf_path, "wb") as f:
                    f.write(response.content)
        target_pdf_path = local_pdf_path
    except Exception as e:
        st.error(f"Failed to download default PDF: {e}")

st.markdown("---")

# Execution Checks
if not api_key_input:
    st.warning("👈 Please paste your free Groq API key in the sidebar to proceed. (Get one free at console.groq.com)")
elif not target_pdf_path:
    st.info("Please upload a PDF file or select the default textbook option above to begin.")
else:
    @st.cache_resource
    def load_vector_db(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(chunks, embeddings)

    try:
        with st.spinner("Processing document chunks... Please wait."):
            vector_db = load_vector_db(target_pdf_path)

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        st.markdown("### 💬 Step 2: Ask Technical Questions")
        query = st.text_input("Enter your electrical engineering question:")
        
        if query:
            with st.spinner("Searching standards and generating answer..."):
                docs_found = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs_found])
                prompt = f"Context: {context}\n\nQuestion: {query}"
                response = llm.invoke(prompt)
                st.markdown("### Answer:")
                st.write(response.content)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
