from transformers import pipeline

def get_summarizer(model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    return summarizer

def summarize_text(text, summarizer, max_length=150, min_length=30):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summaries = []
    for chunk in chunks:
        input_len = len(summarizer.tokenizer.encode(chunk))
        dynamic_max = min(int(input_len * 0.6), max_length)
        dynamic_max = max(dynamic_max, min_length + 5)
        summary = summarizer(
            chunk,
            max_length=dynamic_max,
            min_length=min_length,
            do_sample=False
        )
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)
