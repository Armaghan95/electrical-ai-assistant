import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI Configuration
st.set_page_config(
    page_title="Electrical Engineering AI Assistant", page_icon="⚡"
)
st.title("⚡ Electrical Engineering AI Assistant")
st.write("Ask technical questions based on your loaded electrical manuals.")

# Note: Ensure your OpenAI API key is set in your environment variables, e.g.:
# export OPENAI_API_KEY="your-api-key"
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    st.warning(
        "Please set your OPENAI_API_KEY environment variable to proceed."
    )
else:
    # Load and process document (Checking if file exists)
    pdf_path = "electrical_manual.pdf"

    if os.path.exists(pdf_path):
        @st.cache_resource
        def load_vector_db():
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            return Chroma.from_documents(chunks, embeddings)

        vector_db = load_vector_db()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # User Query Input
        query = st.text_input("Enter your electrical engineering question:")

        if query:
            with st.spinner("Searching standards and calculating..."):
                docs_found = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs_found])

                prompt = f"""You are an expert electrical engineer assistant. 
                Answer the question based only on the provided context.
                
                Context:
                {context}
                
                Question: {query}
                """
                response = llm.invoke(prompt)
                st.markdown("### Answer:")
                st.write(response.content)
    else:
        st.error(
            f"Could not find '{pdf_path}' in the folder. Please add a sample PDF document to test!"
        )
