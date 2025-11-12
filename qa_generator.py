import streamlit as st
from transformers import pipeline
from transformers import T5Tokenizer

@st.cache_resource
def get_qg_pipeline():
    """
    Loads and caches the Question Generation pipeline.
    We are now explicitly loading the 'slow' T5Tokenizer
    to bypass the fast tokenizer conversion bug.
    """
    model_name = "valhalla/t5-base-qg-hl"
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    return pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)

@st.cache_resource
def get_qa_pipeline():
    """
    Loads and caches the Question Answering pipeline.
    """
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def get_faqs(text):
    """
    Generates a list of question-answer pairs from the text.
    """
    qg_pipeline = get_qg_pipeline()
    qa_pipeline = get_qa_pipeline()
    
    text_chunk = text[:2000]
    
    generated_questions = qg_pipeline(
        f"generate_questions: {text_chunk}", 
        num_return_sequences=5, 
        num_beams=5,            
        num_beam_groups=5,      
        diversity_penalty=1.0   
    )
    
    faqs = []
    
    for item in generated_questions:
        question = item['generated_text']
        try:
            answer = qa_pipeline(question=question, context=text_chunk)
            
            if answer['score'] > 0.1:
                faqs.append({
                    "question": question,
                    "answer": answer['answer']
                })
        except Exception as e:
            print(f"Error processing QA for question '{question}': {e}")
            
    return faqs[:5]