import sys
import os
from summarizer.extract_pdf import extract_text_from_pdf
from summarizer.extract_url import extract_text_from_url
from summarizer.summarize import get_summarizer, summarize_text
from summarizer.preprocess import clean_text

def main(input_path_or_url, is_pdf=True, output_path="outputs/summary.txt"):
    if is_pdf:
        text = extract_text_from_pdf(input_path_or_url)
    else:
        text = extract_text_from_url(input_path_or_url)
    # Optional: Clean text
    text = clean_text(text)
    # Load summarizer
    summarizer = get_summarizer(model_name="facebook/bart-large-cnn")  # Replace with biomedical model name if available
    summary = summarize_text(text, summarizer)
    os.makedirs("outputs", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print("Summary saved to", output_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <file_or_url> <is_pdf: 1 or 0>")
        sys.exit(1)
    input_path_or_url = sys.argv[1]
    is_pdf = bool(int(sys.argv[2]))
    main(input_path_or_url, is_pdf)
