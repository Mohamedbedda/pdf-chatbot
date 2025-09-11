# 💬​ PDF Chatbot

This project is a **Retrieval-Augmented Generation (RAG) chatbot** built with **Google Colab, Gradio, FAISS, and Hugging Face Transformers**.  
It allows you to upload a PDF and ask questions about its content. The system retrieves the most relevant chunks using FAISS and reranks them with a cross-encoder before generating an answer with a large language model (Flan-T5).  

## 🚀 Features
- Upload any PDF and extract its text (via PyMuPDF).  
- Split text into chunks with overlap for better retrieval.  
- Build a FAISS index of embeddings ([all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)).  
- Rerank retrieved passages with a CrossEncoder ([ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)).  
- Generate answers using [Flan-T5](https://huggingface.co/google/flan-t5-large).  
- Simple Gradio interface for interaction.  

## 🛠 Tech Stack
- [Gradio](https://gradio.app/) – Web interface  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector search  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) – LLM & embeddings  
- [Sentence-Transformers](https://www.sbert.net/) – Embeddings  
- [PyMuPDF](https://pymupdf.readthedocs.io/) – PDF text extraction  

## ▶️ Usage
1.  Clone the repository:
  ```bash
    git clone https://github.com/your-username/pdf-chatbot.git
    cd pdf-chatbot
  ```
2.  Open the notebook main.
3.  Run all cells of the notebook. At the end of execution, Colab will display a Gradio public URL.
4.  Click on the link and the chatbot interface will open in your browser.
