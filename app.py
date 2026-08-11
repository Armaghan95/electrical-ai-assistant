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
st.write("Ask technical questions based on your loaded electrical manuals or textbook URLs.")

# Sidebar Configuration for Source Selection
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter Free Groq API Key:", type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

st.sidebar.markdown("---")
st.sidebar.subheader("Choose Document Source")
source_option = st.sidebar.radio("Select input method:", ["PDF URL", "Upload PDF File"])

target_pdf_path = None

if source_option == "PDF URL":
    url_input = st.sidebar.text_input(
        "Enter PDF URL:",
        value="https://mycollegevcampus.com/sjcet/notes/Text_Book_2_Electric_Machinery_And_Power_System_Fundamentals_-_Chapman__S.J..pdf"
    )
    if url_input:
        local_pdf_path = "downloaded_textbook.pdf"
        try:
            response = requests.get(url_input)
            with open(local_pdf_path, "wb") as f:
                f.write(response.content)
            target_pdf_path = local_pdf_path
        except Exception as e:
            st.sidebar.error(f"Failed to download PDF from URL: {e}")

else:
    uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])
    if uploaded_file is not None:
        local_pdf_path = "uploaded_manual.pdf"
        with open(local_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        target_pdf_path = local_pdf_path

# Main Execution Logic
if not api_key_input:
    st.warning("👈 Please paste your free Groq API key in the sidebar to proceed.")
elif not target_pdf_path:
    st.info("👈 Please provide a valid PDF URL or upload a PDF file via the sidebar to begin.")
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

        query = st.text_input("Ask a question about your document:")
        if query:
            with st.spinner("Searching for answers..."):
                docs_found = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs_found])
                prompt = f"Context: {context}\n\nQuestion: {query}"
                response = llm.invoke(prompt)
                st.markdown("### Answer:")
                st.write(response.content)
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
