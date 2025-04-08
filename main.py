import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
import os
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')




# Initialize session state
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.texts = []
    st.session_state.metadata = []

# File uploader
st.title("Multi-PDF Q&A Chat with Source Traceability")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Paragraph-based chunking function
def extract_paragraph_chunks(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    pdf_name = pdf_file.name
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text().strip()
        paragraphs = re.split(r'\n{2,}|\r\n{2,}', text)
        for para in paragraphs:
            clean_para = para.strip()
            if len(clean_para) > 30:
                chunks.append({
                    "text": clean_para,
                    "metadata": {
                        "pdf_name": pdf_name,
                        "page_number": page_num + 1
                    }
                })
    return chunks

# Process PDFs and build FAISS index
if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        all_chunks.extend(extract_paragraph_chunks(file))

    texts = [chunk["text"] for chunk in all_chunks]
    metadata = [chunk["metadata"] for chunk in all_chunks]
    embeddings = embed_model.encode(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.session_state.faiss_index = index
    st.session_state.texts = texts
    st.session_state.metadata = metadata

# Secure input for Groq API key
groq_api_key = st.text_input("Enter your Groq API key", type="password")

# User query input
query = st.text_input("Ask a question about the uploaded PDFs")

# Perform retrieval and generate answer
if query and st.session_state.faiss_index:
    if not groq_api_key:
        st.warning("Please enter your Groq API key to generate an answer.")
    else:
        query_embedding = embed_model.encode([query])
        D, I = st.session_state.faiss_index.search(np.array(query_embedding), k=3)

        top_chunks = []
        for i in I[0]:
            top_chunks.append({
                "text": st.session_state.texts[i],
                "pdf_name": st.session_state.metadata[i]['pdf_name'],
                "page_number": st.session_state.metadata[i]['page_number']
            })

        llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)
        context = "\n\n".join([chunk["text"] for chunk in top_chunks])
        prompt = f"Answer the question based on the context below:\n\n{context}\n\nQuestion: {query}"
        answer = llm.invoke(prompt)

        # Display answer
        st.subheader("Answer")
        st.write(answer)

        # Display source info
        st.subheader("Source Information")
        st.write("Most Matched context is on the top: ")
        for chunk in top_chunks:
            st.markdown(f"- PDF: {chunk['pdf_name']} | Page: {chunk['page_number']}")
