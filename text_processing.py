import requests
from bs4 import BeautifulSoup
import pdfplumber

def get_text_from_url(url):
    """
    Fetches and extracts plain text content from a given URL.
    """
    try:
        r = requests.get(url)
        r.raise_for_status()  
        soup = BeautifulSoup(r.text, 'html.parser')
        paragraphs = soup.find_all('p')
        
        if not paragraphs:
            text = soup.get_text(separator=' ', strip=True)
        else:
            text = " ".join([p.text for p in paragraphs])
            
        return text
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None

def get_text_from_pdf(pdf_file):
    """
    Extracts plain text from an uploaded PDF file.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None