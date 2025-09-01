# Earnings Call Transcript Q&A

An interactive **Streamlit app** to upload, analyze, and query company earnings call transcripts.  
The app uses **LangChain, Hugging Face Transformers, and FAISS** for semantic search and Q&A.  
It also supports **sentiment analysis, keyword highlighting, and microphone input** for asking questions.


## Features
- Upload **TXT/CSV transcripts**
- **Semantic search** powered by FAISS + HuggingFace embeddings
- **Q&A system** using HuggingFace `flan-t5-base`
- **Sentiment analysis** of retrieved snippets
- **Keyword highlighting** in context
- Ask questions by **typing or speaking** (via microphone 🎤)
- Download Q&A results as **CSV**


## 🧠 Tech Stack

- **Frontend / UI**
  - [Streamlit](https://streamlit.io/) → For building the interactive web app
  - [streamlit-mic-recorder](https://pypi.org/project/streamlit-mic-recorder/) → Microphone input support

- **Natural Language Processing**
  - [Hugging Face Transformers](https://huggingface.co/transformers/) → Embeddings & LLM-based Q&A
  - [LangChain](https://www.langchain.com/) → Framework for managing prompts, vector search, and LLM chains
  - [TextBlob](https://textblob.readthedocs.io/) → Sentiment analysis

- **Vector Database / Search**
  - [FAISS](https://github.com/facebookresearch/faiss) → Efficient similarity search on transcript embeddings

- **Backend / Data Handling**
  - [Python](https://www.python.org/) (3.10+)  
  - [Pandas](https://pandas.pydata.org/) → Data loading and manipulation

- **Deployment**
  - [Streamlit Cloud](https://streamlit.io/cloud) → One-click hosting & sharing
  - [GitHub](https://github.com/) → Version control and project hosting



##  Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Esha-26-reddy/earnings-call-rag.git
   cd earnings-call-rag

2. **Create a virtual environment**

python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

3. **Install dependencies**

```pip install -r requirements.txt```

4. **Run the Streamlit app**

```python -m streamlit run app.py```


**Project Structure**

earnings-call-rag/
│── app.py                 # Main Streamlit app
│── build_index.py         # Build FAISS index from transcript
│── query_index.py         # Q&A, sentiment, keyword highlighting
│── requirements.txt       # Python dependencies
│── README.md              # Documentation
