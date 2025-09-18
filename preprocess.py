import re

def clean_text(text):
    # Remove newlines and excess whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove references, if needed
    return text.strip()
