import streamlit as st
from transformers import pipeline
from transformers import T5Tokenizer
import math # Needed for sliding window logic

# Constants for QA chunking (based on distilbert-base-cased-distilled-squad)
MAX_CONTEXT_LENGTH = 384
OVERLAP = 128
QA_QUESTION_LIMIT = 5 

@st.cache_resource
def get_qg_pipeline():
    """
    Loads and caches the Question Generation pipeline.
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
    Generates a list of question-answer pairs from the text using a 
    question generation model and a sliding window for QA over the full text.
    """
    qg_pipeline = get_qg_pipeline()
    qa_pipeline = get_qa_pipeline()
    
    # 1. Generate questions from the first part of the text
    qg_context = text[:2000] 
    
    generated_questions = qg_pipeline(
        f"generate_questions: {qg_context}", 
        num_return_sequences=QA_QUESTION_LIMIT, 
        num_beams=5,            
        num_beam_groups=5,      
        diversity_penalty=1.0   
    )
    
    questions = [item['generated_text'] for item in generated_questions]
    faqs = []
    
    # 2. Answer questions using the full text via Sliding Window (QA)
    tokenizer = qa_pipeline.tokenizer
    full_tokens = tokenizer.encode(text)
    
    total_tokens = len(full_tokens)
    step = MAX_CONTEXT_LENGTH - OVERLAP
    num_chunks = math.ceil(total_tokens / step)
    
    
    for question in questions:
        best_answer = None
        best_score = -1.0
        
        for i in range(num_chunks):
            start = i * step
            end = min(start + MAX_CONTEXT_LENGTH, total_tokens)
            
            chunk_tokens = full_tokens[start:end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            try:
                # Run QA on the current chunk
                current_answer = qa_pipeline(question=question, context=chunk_text)
                
                # Compare scores and keep the best one
                if current_answer['score'] > best_score:
                    best_score = current_answer['score']
                    best_answer = current_answer
                    
            except Exception as e:
                print(f"Error processing QA for question '{question}' in chunk {i}: {e}")

        # 3. Compile the final FAQs based on the best answer found
        if best_answer and best_score > 0.1: # Use a confidence threshold
            faqs.append({
                "question": question,
                "answer": best_answer['answer']
            })
            
    return faqs[:5]