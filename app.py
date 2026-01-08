import streamlit as st
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.abspath('src'))
from rag_pipeline import build_rag_chain

# --- Page Configuration ---
st.set_page_config(page_title="CrediTrust Complaint Analyst", page_icon="🏦")

st.title("🏦 CrediTrust Intelligent Complaint Analysis")
st.markdown("Ask questions about customer feedback across Credit Cards, Loans, and more.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Config & Actions ---
with st.sidebar:
    st.header("Configuration")
    
    # 1. API Key Input
    api_key = st.text_input("Enter HuggingFace API Token", type="password")
    if api_key:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
    
    st.divider()
    
    # 2. Clear Conversation Button (New Feature)
    def clear_chat_history():
        st.session_state.messages = []
    
    if st.button("🗑️ Clear Conversation", type="primary"):
        clear_chat_history()

    if not api_key:
        st.warning("Please enter your HuggingFace API Token to proceed.")
        st.stop()

# --- Load RAG Chain (Cached) ---
# We use cache_resource so the model doesn't reload when you clear chat or interact
@st.cache_resource
def get_chain(api_key):
    return build_rag_chain(api_key)

try:
    qa_chain = get_chain(api_key)
except Exception as e:
    st.error(f"Failed to load RAG pipeline: {e}")
    st.stop()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ex: Why are customers closing their savings accounts?"):
    # 1. Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing complaints..."):
            try:
                # Invoke the RAG chain
                response = qa_chain.invoke({"query": prompt})
                answer = response['result']
                source_docs = response['source_documents']
                
                # Display Answer
                st.markdown(answer)
                
                # Display Sources (Expandable)
                with st.expander("View Evidence (Source Complaints)"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Source {i+1} ({doc.metadata.get('product_category', 'Unknown')}):**")
                        st.markdown(f"> *{doc.page_content}*")
                        st.divider()
                
                # 3. Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")