import streamlit as st

import text_processing
import summarizer
import qa_generator
import article_finder
import visuals

st.set_page_config(
    page_title="BioMedical Article Summarizer",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 BioMedical Article Summarizer")
st.markdown("Upload a PDF or enter a URL to summarize, analyze, and explore biomedical articles.")

with st.sidebar:
    st.header("1. Input Article")
    input_type = st.radio("Select input type:", ("URL", "PDF"))
    
    article_url = None
    uploaded_pdf = None
    
    if input_type == "URL":
        article_url = st.text_input("Enter the article URL:")
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF file:", type=["pdf"])

    st.header("2. Choose Summary Type")
    summary_type = st.radio("Select summary type:", ("Extractive", "Abstractive"))
    
    process_button = st.button("Summarize and Analyze")
    
if process_button:
    article_text = None
    with st.spinner("Step 1/5: Extracting text from source..."):
        try:
            if article_url:
                article_text = text_processing.get_text_from_url(article_url)
            elif uploaded_pdf:
                article_text = text_processing.get_text_from_pdf(uploaded_pdf)
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            st.stop()
            
    if not article_text:
        st.error("Could not extract text. Please check the URL or PDF file.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Visuals", "FAQs", "Related Articles"])

    with tab1:
        st.header(f"{summary_type} Summary")
        with st.spinner(f"Step 2/5: Generating {summary_type.lower()} summary..."):
            if summary_type == "Extractive":
                summary = summarizer.get_extractive_summary(article_text)
            else:
                summary = summarizer.get_abstractive_summary(article_text)
            st.success("Summary Generated!")
            st.write(summary)

    with tab2:
        st.header("Text Visuals")
        with st.spinner("Step 3/5: Creating visuals..."):
            # Word Cloud
            st.subheader("Article Word Cloud")
            wc_fig = visuals.create_wordcloud(article_text)
            if wc_fig:
                st.pyplot(wc_fig)
            else:
                st.write("Could not generate word cloud.")
            
            # NER Chart
            st.subheader("Named Entity Recognition")
            ner_fig = visuals.create_ner_chart(article_text)
            if ner_fig:
                st.plotly_chart(ner_fig, use_container_width=True)
            else:
                st.write("Could not generate NER chart.")

    with tab3:
        st.header("Frequently Asked Questions")
        with st.spinner("Step 4/5: Generating FAQs..."):
            faqs = qa_generator.get_faqs(article_text)
            if faqs:
                for i, faq in enumerate(faqs):
                    with st.expander(f"Q{i+1}: {faq['question']}"):
                        st.write(f"**Answer:** {faq['answer']}")
            else:
                st.write("Could not generate FAQs for this article.")

    with tab4:
        st.header("Related Articles from PubMed")
        with st.spinner("Step 5/5: Finding related articles..."):
            related_articles = article_finder.get_related_articles(article_text)
            if related_articles:
                for article in related_articles:
                    st.markdown(f"**[{article['title']}]({article['url']})**")
            else:
                st.write("Could not find related articles.")

    st.success("Analysis Complete!")
    st.balloons()