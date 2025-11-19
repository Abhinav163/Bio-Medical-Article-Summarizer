import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import scispacy 
import plotly.express as px
import pandas as pd
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize 
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

BIOMEDICAL_MODEL = "en_core_sci_lg" 

@st.cache_resource
def load_spacy_model_for_ner():
    """
    Loads and caches the biomedical spaCy model (SciSpacy).
    """
    try:
        return spacy.load(BIOMEDICAL_MODEL) 
    except OSError:
        st.error(f"SciSpacy model '{BIOMEDICAL_MODEL}' not found. Falling back to 'en_core_web_sm'. Please ensure '{BIOMEDICAL_MODEL}' is installed.")
        return spacy.load("en_core_web_sm") 

def create_wordcloud(text):
    """
    Generates a Matplotlib figure for a word cloud.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        return fig
    except Exception as e:
        print(f"Error creating wordcloud: {e}")
        return None

def create_ner_chart(text):
    """
    Generates a Plotly Bar Chart for Named Entities.
    """
    nlp = load_spacy_model_for_ner()
    doc = nlp(text)
    
    entity_labels = [ent.label_ for ent in doc.ents]
    
    if not entity_labels:
        return None
        
    df = pd.DataFrame(entity_labels, columns=["Entity Label"])
    entity_counts = df['Entity Label'].value_counts().reset_index()
    entity_counts.columns = ['Entity Label', 'Count']
    
    fig = px.bar(entity_counts, 
                x='Entity Label', 
                y='Count', 
                title="Frequency of Biomedical Named Entities (SciSpacy)", 
                color='Entity Label')
    return fig

def create_ngram_chart(text, n=2, top_k=10):
    """
    Generates a Plotly Bar Chart for the frequency of top N-grams (Bi-grams or Tri-grams).
    """
    try:
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 1]
        
        ngram_list = ngrams(filtered_words, n)
        
        ngram_freq = Counter(ngram_list)
        
        top_ngrams = ngram_freq.most_common(top_k)
        
        if not top_ngrams:
            return None
            
        df = pd.DataFrame(top_ngrams, columns=['Phrase Tuple', 'Count'])
        df['Phrase'] = df['Phrase Tuple'].apply(lambda x: ' '.join(x))
        
        fig = px.bar(df, 
                    x='Phrase', 
                    y='Count', 
                    title=f"Top {top_k} Most Frequent {n}-gram Phrases",
                    color='Phrase')
        return fig
    except Exception as e:
        print(f"Error creating N-gram chart: {e}")
        return None

def create_sentence_clustering_plot(text):
    """
    Generates a Plotly Scatter Plot showing sentence similarity using PCA on TF-IDF vectors.
    """
    try:
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 5]
        
        if len(sentences) < 5:
            return None 
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)
        n_components = min(2, tfidf_matrix.shape[1]) 
        
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(tfidf_matrix.toarray())
        
        df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'][:n_components])
        df['Sentence'] = sentences
        df['Index'] = range(len(sentences))
        
        # 3. Plotting
        fig = px.scatter(df, 
                        x='PC1', 
                        y='PC2', 
                        hover_name='Sentence', 
                        title='Sentence Similarity Clustering (PCA on TF-IDF)',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'})
        
        return fig
    except Exception as e:
        print(f"Error creating sentence clustering chart: {e}")
        return None