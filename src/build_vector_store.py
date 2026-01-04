import pandas as pd
import os
from sklearn.model_selection import train_test_split
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

# Configuration
INPUT_PATH = 'data/processed/filtered_complaints.csv'
VECTOR_STORE_DIR = 'vector_store/chroma_db'
SAMPLE_SIZE = 10000 # As per instructions (10k-15k)

def create_stratified_sample(df):
    print(f"Total dataset size: {len(df)}")
    
    # Stratified sample to ensure all products are represented
    # We use train_test_split as a quick hack for sampling
    _, sample_df = train_test_split(
        df, 
        test_size=SAMPLE_SIZE, 
        stratify=df['Standardized_Product'],
        random_state=42
    )
    
    print(f"Sample size: {len(sample_df)}")
    print(sample_df['Standardized_Product'].value_counts())
    return sample_df

def chunk_and_vectorize(df):
    # 1. Define Embedding Model
    print("Loading Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Define Text Splitter
    # 500 chars per chunk, 50 overlap (matches the pre-built dataset specs)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 3. Prepare Documents
    print("Chunking documents...")
    documents = []
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        original_text = row['cleaned_narrative']
        chunks = text_splitter.split_text(original_text)
        
        for i, chunk in enumerate(chunks):
            # Create metadata dictionary
            meta = {
                "complaint_id": str(row['Complaint ID']),
                "product_category": row['Standardized_Product'],
                "issue": row['Issue'] if pd.notna(row['Issue']) else "Unknown",
                "company": row['Company'] if pd.notna(row['Company']) else "Unknown",
                "chunk_index": i
            }
            
            doc = Document(page_content=chunk, metadata=meta)
            documents.append(doc)
            
    print(f"Generated {len(documents)} text chunks.")

    # 4. Create and Persist Vector Store
    # We process in batches to avoid memory issues if dataset grows
    print("Creating Vector Store (ChromaDB)...")
    
    if os.path.exists(VECTOR_STORE_DIR):
        print("Warning: Vector store directory already exists. Appending or Overwriting...")

    # Initialize Chroma
    vector_store = Chroma(
        collection_name="complaints_rag",
        embedding_function=embedding_model,
        persist_directory=VECTOR_STORE_DIR
    )

    # Add documents in batches of 5000 (Chroma handles this well, but explicit batching is safe)
    batch_size = 5000
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vector_store.add_documents(documents=batch)
        print(f"Processed batch {i} to {i+batch_size}")

    print(f"Vector store successfully saved to {VECTOR_STORE_DIR}")

def main():
    if not os.path.exists(INPUT_PATH):
        print("Processed data not found. Run Task 1 first.")
        return

    df = pd.read_csv(INPUT_PATH)
    
    # 1. Sample
    sample_df = create_stratified_sample(df)
    
    # 2. Chunk & Embed
    chunk_and_vectorize(sample_df)

if __name__ == "__main__":
    main()