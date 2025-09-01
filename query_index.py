from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from textblob import TextBlob
import re


# 1. Load LLM

def get_llm(model_name="google/flan-t5-base", device=-1):
    """
    Load a HuggingFacePipeline-wrapped LLM for text2text generation.
    """
    pipe = pipeline("text2text-generation", model=model_name, device=device)
    return HuggingFacePipeline(pipeline=pipe)

# 2. Retrieval QA

def run_qa(llm, db, query, k=3):
    """
    Run retrieval-augmented QA using the provided LLM and FAISS DB.
    Returns answer and source documents.
    """
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    result = qa.invoke({"query": query})
    return result


# 3. Speaker Diarization

def diarize_transcript(filepath, num_speakers=2, skip=False):
    """
    Simple diarization placeholder.
    Alternates lines between speakers or skips if skip=True.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if skip:
        return "".join(lines)
    
    speakered = []
    for i, line in enumerate(lines):
        if line.strip():
            speakered.append(f"[Speaker {(i % num_speakers)+1}] {line.strip()}")
    return "\n".join(speakered)


# 4. Keyword Highlighting

def highlight_keywords(text, query):
    """
    Highlight whole-word query terms in text using markdown bold.
    """
    for word in query.split():
        regex = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        text = regex.sub(f"**{word}**", text)
    return text

# 5. Sentiment Analysis

def analyze_sentiment(texts):
    """
    Simple sentiment analysis using TextBlob.
    Returns list of dicts with label and normalized score (0-1).
    """
    results = []
    for t in texts:
        blob = TextBlob(t)
        polarity = blob.sentiment.polarity  # -1 to 1
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        # normalize score 0-1 for easier charting
        score = (polarity + 1) / 2
        results.append({"label": label, "score": score})
    return results
