import streamlit as st
from transformers import pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import nltk

nltk.download('punkt', quiet=True)

@st.cache_resource
def get_abstractive_pipeline():
    """
    Loads and caches the abstractive summarization pipeline.
    """
    return pipeline("summarization", model="facebook/bart-large-cnn")

def get_abstractive_summary(text):
    """
    Generates an abstractive summary using Hugging Face Transformers.
    Handles text chunking for long articles.
    """
    summarizer = get_abstractive_pipeline()
    
    max_chunk_length = 1024
    chunks = [text[i:i + max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    summary_text = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary_text += result[0]['summary_text'] + " "
        
    return summary_text.strip()


def get_extractive_summary(text, sentences_count=7):
    """
    Generates an extractive summary using Sumy (TextRank).
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    
    return " ".join([str(sentence) for sentence in summary])