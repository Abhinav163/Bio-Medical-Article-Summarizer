import re
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words = set(stopwords.words('english'))

def preprocess_sentence(sentence):
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

def extractive_summarize(text, top_n=5):
    sentences = sent_tokenize(text)
    if len(sentences) <= top_n:
        return " ".join(sentences)

    clean_sentences = [preprocess_sentence(s).lower() for s in sentences]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(clean_sentences)

    sim_matrix = cosine_similarity(tfidf_matrix)

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    selected_sentences = [s for _, s in ranked_sentences[:top_n]]
    summary = " ".join(selected_sentences)
    return summary
