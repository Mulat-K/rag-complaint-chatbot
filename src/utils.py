import re

def clean_text(text):
    """
    Cleans complaint narrative text.
    - Lowercases text
    - Removes redactions (XXXX)
    - Removes special characters
    - Normalizes whitespace
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove XXXX (CFPB redactions)
    text = re.sub(r'x{2,}', '', text)
    
    # Remove special characters but keep punctuation useful for sentence splitting
    text = re.sub(r'[^a-z0-9\s.,?!]', '', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text