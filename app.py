import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Streamlit UI Configuration
st.set_page_config(
    page_title="Electrical Engineering AI Assistant", page_icon="⚡"
)
st.title("⚡ Electrical Engineering AI Assistant")
st.write(
    "Ask technical questions based on your loaded electrical manuals (IEC"
    " standards)."
)

# Sidebar for safe API key input
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input(
    "Enter OpenAI API Key:", type="password", key="user_api_key"
)

if api_key_input:
  os.environ["OPENAI_API_KEY"] = api_key_input

api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
  st.warning("👈 Please paste your OpenAI API key in the sidebar to proceed.")
else:
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

    try:
      with st.spinner(
          "Processing electrical manual and embedding chunks... Please wait."
      ):
        vector_db = load_vector_db()

      retriever = vector_db.as_retriever(search_kwargs={"k": 3})
      llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

      query = st.text_input("Enter your electrical engineering question:")

      if query:
        with st.spinner("Searching standards and generating answer..."):
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

    except Exception as e:
      st.error(
          f"An error occurred (Check if your API key is valid or if you have"
          f" active credits): {e}"
      )
  else:
    st.error(
        f"Could not find '{pdf_path}' in your project folder. Make sure your PDF"
        " is placed next to app.py and named correctly."
    )
