import streamlit as st
from transformers import pipeline, AutoTokenizer # NEW: Added AutoTokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

nltk.download('punkt', quiet=True)

MODEL_NAME = "facebook/bart-large-cnn"
MAX_TOKENS = 1000 # Use 1000 tokens for safety margin (BART max is 1024)

@st.cache_resource
def get_abstractive_pipeline():
    """
    Loads and caches the abstractive summarization pipeline.
    """
    return pipeline("summarization", model=MODEL_NAME)

@st.cache_resource
def get_bart_tokenizer():
    """
    Loads and caches the BART tokenizer.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_abstractive_summary(text, max_length=200, min_length=30):
    """
    Generates an abstractive summary using Hugging Face Transformers.
    Implements a two-step (hierarchical/meta-document) approach for long articles,
    with safe token-based chunking and truncation.
    """
    summarizer = get_abstractive_pipeline()
    tokenizer = get_bart_tokenizer()
    
    # 1. First Pass: Summarize in token-based chunks
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=False)
    total_length = input_ids.size(1)
    
    meta_summary_list = []
    
    # Iterate through token chunks
    for i in range(0, total_length, MAX_TOKENS):
        chunk_ids = input_ids[0, i:i + MAX_TOKENS]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        if not chunk_text.strip():
            continue
            
        # Use a consistent, shorter length for chunk summaries
        chunk_result = summarizer(chunk_text, max_length=100, min_length=20, do_sample=False)
        meta_summary_list.append(chunk_result[0]['summary_text'])
        
    meta_document = " ".join(meta_summary_list)
    
    if not meta_document.strip():
        return "" 
        
    # 2. Second Pass: Summarize the Meta-Document
    
    # CRITICAL FIX: Truncate the meta-document input before the final call
    final_input_ids = tokenizer.encode(
        meta_document, 
        return_tensors='pt', 
        truncation=True, 
        max_length=MAX_TOKENS # Ensure input is below 1024 tokens
    )
    final_input_text = tokenizer.decode(final_input_ids[0], skip_special_tokens=True)
    
    final_result = summarizer(
        final_input_text, # Use the safely truncated text
        max_length=max_length, 
        min_length=min_length, 
        do_sample=False
    )
        
    return final_result[0]['summary_text'].strip()


def get_extractive_summary(text, sentences_count=7, algorithm="TextRank"):
    """
    Generates an extractive summary using Sumy (TextRank, LSA).
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    if algorithm == "TextRank":
        summarizer = TextRankSummarizer()
    elif algorithm == "LSA":
        summarizer = LsaSummarizer() 
    else:
        summarizer = TextRankSummarizer() 
        
    summary = summarizer(parser.document, sentences_count)
    
    return " ".join([str(sentence) for sentence in summary])