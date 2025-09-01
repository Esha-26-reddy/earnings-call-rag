import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def build_faiss_from_file(filepath, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Build a FAISS DB from a txt or csv transcript file and return the DB."""
    # create embedding instance
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # load file content into a list of langchain Documents
    ext = os.path.splitext(filepath)[1].lower()
    raw_docs = []

    if ext == ".txt":
        # load plain text using TextLoader (returns list[Document])
        loader = TextLoader(filepath, encoding="utf-8")
        raw_docs = loader.load()

    elif ext == ".csv":
        # read CSV and try to detect a text column, else combine rows
        df = pd.read_csv(filepath)
        candidates = [c for c in df.columns if c.lower() in ("transcript", "text", "content", "body", "utterance")]
        if candidates:
            text_col = candidates[0]
            texts = df[text_col].astype(str).tolist()
            combined = "\n\n".join(texts)
            raw_docs = [Document(page_content=combined, metadata={"source": filepath})]
        else:
            rows = df.astype(str).apply(lambda r: " ".join(r.values), axis=1).tolist()
            combined = "\n\n".join(rows)
            raw_docs = [Document(page_content=combined, metadata={"source": filepath})]

    else:
        # fallback: try TextLoader
        loader = TextLoader(filepath, encoding="utf-8")
        raw_docs = loader.load()

    # split into smaller chunks for embeddings
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)

    # build and return FAISS index
    db = FAISS.from_documents(docs, embeddings)
    return db
