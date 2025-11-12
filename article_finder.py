import streamlit as st
from pymed import PubMed
import spacy

@st.cache_resource
def load_spacy_model():
    """
    Loads and caches the spaCy model.
    """
    return spacy.load("en_core_web_sm")

def get_related_articles(text):
    """
    Finds related articles on PubMed based on keywords from the text.
    """
    nlp = load_spacy_model()
    doc = nlp(text[:1000])
    keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    
    if not keywords:
        return []
        
    search_term = " ".join(set(keywords[:10]))
    
    try:
        pubmed = PubMed(tool="BioMedSummarizer", email="your_email@example.com")
        results = pubmed.query(search_term, max_results=5)
        
        articles = []
        for article in results:
            articles.append({
                "title": article.title,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}/"
            })
        return articles
    except Exception as e:
        print(f"Error querying PubMed: {e}")
        return []