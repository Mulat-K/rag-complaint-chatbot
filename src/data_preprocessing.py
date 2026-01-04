import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Configuration
RAW_DATA_PATH = 'data/raw/complaints.csv' # Ensure you download the CSV here
PROCESSED_DATA_PATH = 'data/processed/filtered_complaints.csv'

def load_and_explore_data():
    print("Loading data...")
    # Loading only necessary columns to save memory if file is huge
    cols = ['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 
            'Consumer complaint narrative', 'Company', 'State', 'Complaint ID']
    
    # Check if file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found. Please download the CFPB dataset.")
        return None

    df = pd.read_csv(RAW_DATA_PATH, usecols=cols)
    print(f"Initial Shape: {df.shape}")
    
    # 1. Check Missing Narratives
    missing_narratives = df['Consumer complaint narrative'].isna().sum()
    print(f"Rows without narratives: {missing_narratives}")
    
    # Drop rows without narratives immediately
    df = df.dropna(subset=['Consumer complaint narrative'])
    print(f"Shape after dropping missing narratives: {df.shape}")
    
    return df

def filter_products(df):
    # Mapping raw CFPB categories to CrediTrust categories
    # Note: 'Money Transfers' often appears as 'Money transfer, virtual currency, or money service'
    product_map = {
        'Credit card': 'Credit Card',
        'Credit card or prepaid card': 'Credit Card',
        'Prepaid card': 'Credit Card',
        
        'Payday loan, title loan, or personal loan': 'Personal Loan',
        'Personal loan': 'Personal Loan', # Legacy category
        
        'Checking or savings account': 'Savings Account',
        'Bank account or service': 'Savings Account',
        
        'Money transfer, virtual currency, or money service': 'Money Transfers',
        'Money transfers': 'Money Transfers'
    }
    
    # Apply mapping
    df['Standardized_Product'] = df['Product'].map(product_map)
    
    # Filter only for mapped products
    df_filtered = df.dropna(subset=['Standardized_Product']).copy()
    
    print("\n--- Product Distribution (Filtered) ---")
    print(df_filtered['Standardized_Product'].value_counts())
    
    return df_filtered

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # The CFPB redacts personal info as "XXXX". 
    # We can keep it to maintain sentence structure, or remove it.
    # Let's remove specific boilerplate "I am writing to..." if commonly found,
    # but for now, we will perform basic normalization.
    
    # Remove newlines
    text = re.sub(r'\n', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_lengths(df):
    # Calculate word count
    df['word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True)
    plt.title('Distribution of Complaint Narrative Lengths')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.savefig('notebooks/narrative_length_dist.png')
    print("Length distribution plot saved to notebooks/")
    
    print(f"Average Word Count: {df['word_count'].mean():.2f}")

def main_pipeline():
    df = load_and_explore_data()
    if df is None: return

    df = filter_products(df)
    
    print("Cleaning text...")
    df['cleaned_narrative'] = df['Consumer complaint narrative'].apply(clean_text)
    
    analyze_lengths(df)
    
    # Save processed data
    print(f"Saving {len(df)} records to {PROCESSED_DATA_PATH}...")
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("Task 1 Complete.")

if __name__ == "__main__":
    main_pipeline()