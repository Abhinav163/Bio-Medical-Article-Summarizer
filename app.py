import streamlit as st

import text_processing
import summarizer
import qa_generator
import article_finder
import visuals
import json
from requests.exceptions import RequestException
from pdfminer.pdfdocument import PDFTextExtractionNotAllowed

st.set_page_config(
    page_title="BioMedical Article Summarizer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 BioMedical Article Summarizer")
st.markdown("Upload a PDF or enter a URL to summarize, analyze, and explore biomedical articles.")

with st.sidebar:
    st.header("1. Article Input")
    input_type = st.radio("Select input type:", ("URL", "PDF"))
    
    article_url = None
    uploaded_pdf = None
    
    if input_type == "URL":
        article_url = st.text_input("Enter the article URL:")
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF file:", type=["pdf"])

    st.header("2. Analysis Settings")
    
    summary_type = st.radio(
        "Select summary type:", 
        ("Abstractive (Hierarchical BART)", "Extractive (TextRank)", "Extractive (LSA)") 
    )
    
    if "Abstractive" in summary_type:
        abstractive_max_length = st.slider(
            "Abstractive Max Summary Length (tokens)", 
            min_value=100, max_value=400, value=200, step=10
        )
    else:
        extractive_sentences = st.slider(
            "Extractive Summary Length (sentences)", 
            min_value=3, max_value=15, value=7, step=1
        )
    
    pubmed_max_results = st.slider(
        "Max Related Articles from PubMed", 
        min_value=3, max_value=15, value=5, step=1
    )
    
    process_button = st.button("Summarize and Analyze")
    

if process_button:
    article_text = None
    article_title = "Article Title (Default)" 

    with st.spinner("Step 1/5: Extracting text and title from source..."):
        try:
            if article_url and input_type == "URL":
                article_text, article_title = text_processing.get_text_from_url(article_url)
            elif uploaded_pdf and input_type == "PDF":
                article_text, article_title = text_processing.get_text_from_pdf(uploaded_pdf)
        except RequestException as e:
            st.error(f"Network Error: Could not fetch URL. {e}")
            st.stop()
        except PDFTextExtractionNotAllowed:
            st.error("PDF Error: Text extraction is blocked (protected PDF).")
            st.stop()
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            st.stop()
            
    if not article_text:
        st.error("Could not extract text. Please check the URL or PDF file.")
        st.stop()

    st.header(f"Article: **{article_title}**")
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Visuals", "FAQs (Sliding Window QA)", "Related Articles"])

    with tab1:
        st.subheader(f"{summary_type} Summary")
        with st.spinner(f"Step 2/5: Generating summary..."):
            if "Abstractive" in summary_type:
                summary = summarizer.get_abstractive_summary(article_text, max_length=abstractive_max_length)
            else:
                algorithm = "TextRank" if "TextRank" in summary_type else "LSA"
                summary = summarizer.get_extractive_summary(article_text, sentences_count=extractive_sentences, algorithm=algorithm)
                
            st.success("Summary Generated!")
            st.markdown("### Summary Text")
            st.write(summary)
            
            with st.expander("View Original Article Text"):
                st.text(article_text)


    with tab2:
        st.subheader("Text Visuals")
        st.markdown("A deeper look into the article's keywords and structure.")
        
        st.markdown("#### Article Word Cloud")
        with st.spinner("Creating Word Cloud..."):
            wc_fig = visuals.create_wordcloud(article_text)
            if wc_fig:
                st.pyplot(wc_fig)
            else:
                st.write("Could not generate word cloud.")
        
        st.markdown("#### Biomedical Named Entity Recognition (SciSpacy)")
        with st.spinner("Creating NER Chart..."):
            ner_fig = visuals.create_ner_chart(article_text)
            if ner_fig:
                st.plotly_chart(ner_fig, use_container_width=True)
            else:
                st.write("Could not generate NER chart. Ensure the SciSpacy model is installed.")
                
        st.markdown("#### Top 10 Most Frequent Bi-grams")
        with st.spinner("Creating N-gram Chart..."):
            ngram_fig = visuals.create_ngram_chart(article_text, n=2, top_k=10)
            if ngram_fig:
                st.plotly_chart(ngram_fig, use_container_width=True)
            else:
                st.write("Could not generate N-gram chart (Text may be too short).")
            
        st.markdown("#### Sentence Similarity Map (PCA on TF-IDF)")
        with st.spinner("Creating Sentence Clustering Map..."):
            clustering_fig = visuals.create_sentence_clustering_plot(article_text)
            if clustering_fig:
                st.plotly_chart(clustering_fig, use_container_width=True)
            else:
                st.write("Could not generate sentence clustering map (Text is too short or lacks variance).")


    with tab3:
        st.subheader("Frequently Asked Questions")
        st.markdown("*(Generated by T5-QG model and answered using full-text Sliding Window QA)*")
        with st.spinner("Step 4/5: Generating FAQs..."):
            faqs = qa_generator.get_faqs(article_text)
            if faqs:
                for i, faq in enumerate(faqs):
                    with st.expander(f"Q{i+1}: {faq['question']}"):
                        st.write(f"**Answer:** {faq['answer']}")
            else:
                st.write("Could not generate FAQs for this article. Try a text with distinct sections.")

    with tab4:
        st.subheader("Related Articles from PubMed")
        with st.spinner(f"Step 5/5: Finding {pubmed_max_results} related articles..."):
            # Pass the max_results setting to the function
            related_articles = article_finder.get_related_articles(article_text, max_results=pubmed_max_results)
            if related_articles:
                for article in related_articles:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
            else:
                st.write("Could not find related articles. Try a text with more distinctive keywords.")

    st.success("Analysis Complete!")
    st.balloons()