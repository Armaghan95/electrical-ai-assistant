import os
import streamlit as st
from langchain_community.document_loaders import OnlinePDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI Configuration
st.set_page_config(page_title="Electrical Engineering AI Assistant", page_icon="⚡")
st.title("⚡ Electrical Engineering AI Assistant")
st.write("Using Chapman's Electric Machinery & Power System Fundamentals.")

# Sidebar for Groq API Key
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("Enter Free Groq API Key:", type="password")

if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input

if not api_key_input:
    st.warning("👈 Please paste your free Groq API key in the sidebar to proceed.")
else:
    # URL of your PDF
    pdf_url = "https://mycollegevcampus.com/sjcet/notes/Text_Book_2_Electric_Machinery_And_Power_System_Fundamentals_-_Chapman__S.J..pdf"

    @st.cache_resource
    def load_vector_db():
        # Loads the PDF directly from the URL
        loader = OnlinePDFLoader(pdf_url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(chunks, embeddings)

    try:
        with st.spinner("Downloading and processing the textbook... this may take a moment."):
            vector_db = load_vector_db()

        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        query = st.text_input("Ask a question about the textbook:")
        if query:
            with st.spinner("Searching for answers..."):
                docs_found = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs_found])
                prompt = f"Context: {context}\n\nQuestion: {query}"
                response = llm.invoke(prompt)
                st.markdown("### Answer:")
                st.write(response.content)
    except Exception as e:
        st.error(f"Error: {e}")
