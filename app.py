import streamlit as st
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath('src'))
from rag_pipeline import build_rag_chain

# Page Config
st.set_page_config(page_title="CrediTrust Complaint Analyst", page_icon="🏦")

st.title("🏦 CrediTrust Intelligent Complaint Analysis")
st.markdown("Ask questions about customer feedback across Credit Cards, Loans, and more.")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter HuggingFace API Token", type="password")
    if not api_key:
        st.warning("Please enter your HuggingFace API Token to proceed.")
        st.stop()
    
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key

# Initialize Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize Chain (Cached to prevent reloading on every interaction)
@st.cache_resource
def get_chain(api_key):
    return build_rag_chain(api_key)

try:
    qa_chain = get_chain(api_key)
except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ex: Why are customers closing their savings accounts?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            try:
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                source_docs = response['source_documents']
                
                st.markdown(answer)
                
                # Expandable Sources Section
                with st.expander("View Evidence (Source Complaints)"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1} ({doc.metadata['product_category']}):**")
                        st.markdown(f"> *{doc.page_content}*")
                        st.divider()
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")