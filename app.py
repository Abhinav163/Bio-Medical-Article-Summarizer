import streamlit as st
from extract_pdf import extract_text_from_pdf
from extract_url import extract_text_from_url
from summarize import get_summarizer, summarize_text
from preprocess import clean_text
from extractive import extractive_summarize

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from wordcloud import wordcloud
import matplotlib.pyplot as plt


import tempfile
import os

st.set_page_config(page_title="Biomedical Summarizer", layout="wide")

st.title("🩺 Biomedical Article Summarizer") 
st.write("Upload a **PDF** or paste a **URL** of a biomedical article to get a summary.")

model_options = {
    "facebook/bart-large-cnn": "Facebook BART Large CNN (General)",
    "sshleifer/distilbart-cnn-12-6": "DistilBART CNN (General)",
    "allenai/led-base-16384": "LED Base Longformer (General, long doc)"
}


option_model = st.sidebar.selectbox("Choose Model or Select 'Compare for All'", 
                                    options=["Compare"] + list(model_options.keys()), 
                                    format_func=lambda x: "Compare All" if x=="Compare" else model_options.get(x, x))

summary_type = st.sidebar.radio("Summarization Type", ["Abstractive", "Extractive"])

max_len = st.sidebar.slider("Max Summary Length", 50, 300, 150)
min_len = st.sidebar.slider("Min Summary Length", 10, 100, 30)

@st.cache_resource
def load_summarizer(model_name):
    return get_summarizer(model_name)

# Preload summarizers if compare is selected
summarizers = {}
if option_model == "Compare":
    for m in model_options:
        summarizers[m] = load_summarizer(m)
else:
    summarizers[option_model] = load_summarizer(option_model)

uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
url_input = st.text_input("Or paste article URL here")

if st.button("Summarize"):

    extracted_text = ""

    if uploaded_pdf is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            tmp_file_path = tmp_file.name
            extracted_text = extract_text_from_pdf(tmp_file_path)
            os.remove(tmp_file_path)

    elif url_input.strip() != "":
        extracted_text = extract_text_from_url(url_input.strip())

    else:
        st.warning("Please upload a PDF or enter a URL.")
        st.stop()

    cleaned_text = clean_text(extracted_text)

    if not cleaned_text:
        st.error("No text could be extracted from the provided source.")
        st.stop()

    st.subheader("📄 Generated Summary")

    if option_model == "Compare":
        results = {}
        for model_name, summarizer in summarizers.items():
            if summary_type == "Abstractive":
                summary = summarize_text(cleaned_text, summarizer, max_length=max_len, min_length=min_len)
            else:  # Extractive
                summary = extractive_summarize(cleaned_text, top_n=5)
            results[model_name] = summary

        # Display side by side
        col1, col2, col3 = st.columns(3)
        for col, (model_name, summary) in zip([col1, col2, col3], results.items()):
            col.markdown(f"**{model_options[model_name]}**")
            col.write(summary)
            col.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"summary_{model_name.replace('/', '_')}.txt",
                mime="text/plain"
            )
        # Visualization of summary lengths
        lengths = [len(results[m]) for m in results]
        st.subheader("📊 Summary Length Comparison (characters)")
        st.bar_chart({model_options[m]: lengths[i] for i,m in enumerate(results)})

    else:
        summarizer = summarizers[option_model]
        if summary_type == "Abstractive":
            summary_result = summarize_text(cleaned_text, summarizer, max_length=max_len, min_length=min_len)
        else:
            summary_result = extractive_summarize(cleaned_text, top_n=5)

        st.write(summary_result)
        st.download_button(
            label="Download Summary",
            data=summary_result,
            file_name="summary.txt",
            mime="text/plain"
        )

        # Keyword Cloud Visualization
        st.subheader("🔑 Keyword Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(summary_result)
        plt.figure(figsize=(12,6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)



# import streamlit as st
# from extract_pdf import extract_text_from_pdf
# from extract_url import extract_text_from_url
# from summarize import get_summarizer, summarize_text
# from preprocess import clean_text
# import tempfile
# import os

# # Title
# st.set_page_config(page_title="Biomedical Summarizer", layout="wide")
# st.title("Biomedical Article Summarizer")
# st.write("Upload a **PDF** or paste a **URL** of a biomedical article to get a summary.")

# # Sidebar - Model selection
# # model_name = st.sidebar.text_input("HuggingFace Model Name", "facebook/bart-large-cnn")
# max_len = st.sidebar.slider("Max Summary Length", 50, 300, 150)
# min_len = st.sidebar.slider("Min Summary Length", 10, 100, 30)

# # Initial summarizer load (cached)
# @st.cache_resource
# def load_summarizer(model_name):
#     return get_summarizer(model_name)

# # summarizer = load_summarizer(model_name)

# # PDF Upload
# uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

# # URL Input
# url_input = st.text_input("Or paste article URL here")

# if st.button("Summarize"):
#     extracted_text = ""

#     # If PDF is uploaded
#     if uploaded_pdf is not None:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_pdf.read())
#             tmp_file_path = tmp_file.name
#         extracted_text = extract_text_from_pdf(tmp_file_path)
#         os.remove(tmp_file_path)

#     # If URL is provided
#     elif url_input.strip() != "":
#         extracted_text = extract_text_from_url(url_input.strip())

#     else:
#         st.warning("Please upload a PDF or enter a URL.")
#         st.stop()

#     # Clean & summarize
#     cleaned_text = clean_text(extracted_text)
#     if not cleaned_text:
#         st.error("No text could be extracted from the provided source.")
#         st.stop()

#     with st.spinner("Generating summary..."):
#         summary_result = summarize_text(cleaned_text, summarizer, max_length=max_len, min_length=min_len)

#     st.subheader("Generated Summary")
#     st.write(summary_result)

#     st.download_button(
#         label="Download Summary",
#         data=summary_result,
#         file_name="summary.txt",
#         mime="text/plain"
#     )