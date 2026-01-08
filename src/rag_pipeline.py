import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint

# Configuration
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# You will need a Hugging Face API Token (Free)
# Export it in terminal: export HUGGINGFACEHUB_API_TOKEN=hf_...
# Or pass it directly (not recommended for production)
HF_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2" # Good open source model

def load_retriever():
    """Loads the existing vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_PATH, 
        embedding_function=embeddings
    )
    
    # K=5 means retrieve top 5 relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 5})

def setup_llm(hf_token=None):
    """Sets up the LLM endpoint."""
    if not hf_token:
        # Try to get from environment variable
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set.")

    llm = HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        max_new_tokens=512,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.1,
        huggingfacehub_api_token=hf_token
    )
    return llm

def build_rag_chain(hf_token=None):
    """Combines Retriever and Generator into a Chain."""
    
    retriever = load_retriever()
    llm = setup_llm(hf_token)

    # Custom Prompt
    prompt_template = """
    You are a financial analyst assistant for CrediTrust. 
    Your task is to answer questions about customer complaints. 
    Use the following retrieved complaint excerpts to formulate your answer. 
    If the context doesn't contain the answer, state that you don't have enough information.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    ANSWER:
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True, # Important for source citation
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return chain