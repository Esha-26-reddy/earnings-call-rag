import streamlit as st
st.set_page_config(page_title="Earnings Call Q&A", layout="wide")

import tempfile, os
import pandas as pd
from io import StringIO

from build_index import build_faiss_from_file
from query_index import get_llm, run_qa, analyze_sentiment, diarize_transcript, highlight_keywords

# Optional mic recorder
try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


@st.cache_resource
def load_llm_cached():
    return get_llm()


st.title(" Earnings Call Transcript Q&A")

# --- Upload transcript ---
uploaded_file = st.file_uploader("Upload a transcript file (TXT or CSV)", type=["txt", "csv"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        filepath = tmp_file.name

    try:
        db = build_faiss_from_file(filepath)
        st.success("✅ Transcript indexed successfully!")
        speakered_transcript = diarize_transcript(filepath)
    except Exception as e:
        st.error(f"Failed to process transcript: {e}")
        raise

    llm = load_llm_cached()

    # --- Single unified input (typing + optional mic) ---
    st.subheader("Ask a question (Type or Speak) 🎤")

    # Ensure session state key exists
    if "question" not in st.session_state:
        st.session_state.question = ""

    # Mic recording FIRST → update session_state
    if MIC_AVAILABLE:
        spoken_text = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")
        if spoken_text:
            st.session_state.question = spoken_text

    # Now create the text input bound to session_state
    st.text_input("Your Question", key="question")

    # Final question variable
    question = st.session_state.question

    # --- Run Q&A ---
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("⚠️ Please enter or speak a question.")
        else:
            st.write(f"**Your question:** {question}")
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                result = run_qa(llm, db, question)

            answer = result.get("result") or result.get("answer") or ""
            st.markdown("###  Answer")
            st.write(answer)

            # Show sources
            sources = result.get("source_documents", [])
            if sources:
                st.markdown("###  Sources (retrieved snippets)")
                for i, doc in enumerate(sources, start=1):
                    snippet = doc.page_content.strip().replace("\n", " ")
                    snippet = highlight_keywords(snippet, question)
                    src = getattr(doc, "metadata", {}).get("source", "unknown")
                    st.markdown(
                        f"**{i}. [{src}]** {snippet[:400]}{'...' if len(snippet) > 400 else ''}"
                    )

            # Sentiment analysis
            st.subheader(" Sentiment Analysis")
            sentiments = analyze_sentiment([doc.page_content for doc in sources])
            for i, s in enumerate(sentiments, start=1):
                st.write(f"Snippet {i}: **{s['label']}** (Score: {s['score']:.2f})")

            # Download Q&A
            st.subheader(" Download Q&A")
            df_download = pd.DataFrame([{"Question": question, "Answer": answer}])
            csv_buffer = StringIO()
            df_download.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Q&A as CSV",
                data=csv_buffer.getvalue(),
                file_name="qa_results.csv",
                mime="text/csv",
            )
