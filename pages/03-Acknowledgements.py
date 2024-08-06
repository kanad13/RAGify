import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="Blunder Mifflin", page_icon=":soccer:", layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.write("""
### Acknowledgements

RAGify is powered by a combination of open and closed-source technologies. I am grateful for the contributions of the following initiatives and organizations:

- [Groq API](https://wow.groq.com/why-groq/) - The LLM functionality in RAGify is powered by Groq's API. (Note: [Groq and Grok are different](https://wow.groq.com/welcome-to-groqs-galaxy-elon/).)
- [Meta Llama 3.1 8B Model](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) - This LLM, hosted by Groq, serves as the core engine behind RAGify's intelligent responses.
- [Hugging Face SentenceTransformer](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) -The `all-mpnet-base-v2` model is used to generate embeddings for semantic search within the system.
- [FAISS (Facebook AI Similarity Search)](https://faiss.ai) - FAISS is used for performing similarity searches.
- [PyPDF](https://pypdf.readthedocs.io/en/latest/index.html) - PyPDF is used for text extraction and processing from PDF documents.
- [Langchain's RecursiveCharacterTextSplitter](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) - This tool is used for breaking down large text into manageable chunks, optimizing them for embedding and retrieval.
- [Streamlit](https://streamlit.io/) - Streamlit provides the platform for hosting RAGify - a user-friendly web interface.
- [Python](https://github.com/python) - The backbone of RAGify's codebase.
- The Broader AI/ML Community - A special thanks to the AI/ML community whose ongoing research and open-source contributions have laid the foundation for this project.
				 """)
