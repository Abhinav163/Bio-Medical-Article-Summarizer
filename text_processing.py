import requests
import pdfplumber
import trafilatura
import json # <-- NEW: Import the standard JSON library
# Import specific exception types for better error handling
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed
from requests.exceptions import RequestException

def get_text_from_url(url):
    """
    Fetches and extracts plain text content and title from a given URL using trafilatura.
    Returns: (text, title)
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise RequestException(f"Could not download content from URL: {url}")
            
        # Extract metadata and text as a JSON string
        extracted_content = trafilatura.extract(
            downloaded, 
            include_comments=False, 
            include_tables=False,
            output_format='json'
        )
        
        if extracted_content:
            # FIX: Use json.loads() instead of the incorrect trafilatura.json_to_dict()
            data = json.loads(extracted_content) 
            text = data.get('text', '')
            title = data.get('title', 'Article Title (Extracted by Trafilatura)')
            if not text:
                 raise Exception("Could not extract main article text.")
            return text, title
        
        # Fallback if trafilatura.extract didn't return content (shouldn't happen with 'json' output)
        text = trafilatura.extract(downloaded)
        if text:
             return text, url # Use URL as placeholder title
        
        raise Exception("Could not extract main article text.")
            
    except RequestException as e:
        print(f"Error fetching URL: {e}")
        return None, None
    except Exception as e:
        print(f"Error extracting content from URL: {e}")
        return None, None

def get_text_from_pdf(pdf_file):
    """
    Extracts plain text from an uploaded PDF file.
    Returns: (text, title)
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # A PDF doesn't easily provide a title. Use the filename as a fallback.
        title = pdf_file.name.replace(".pdf", "").replace("_", " ").title()
        
        if not text:
            raise Exception("PDF file is empty or text extraction failed.")
            
        return text, title
        
    except PDFTextExtractionNotAllowed:
        print("Error reading PDF: Text extraction is not allowed (protected PDF).")
        return None, None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None, None