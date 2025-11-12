import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
import plotly.express as px
import pandas as pd

@st.cache_resource
def load_spacy_model_for_ner():
    """
    Loads and caches the spaCy model.
    """
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
                title="Frequency of Named Entities",
                color='Entity Label')
    return fig