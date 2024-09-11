import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="RAGify", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### RAGify - Chat with Your Documents using Gen AI

This chatbot is for RAGify - a tool that lets you chat with your documents using Generative AI & Retrieval-Augmented Generation (RAG).

Think of RAGify as giving your documents a brain.

#### Why RAGify?

Individuals and companies often accumulate vast amounts of documentation in various formats like PDF and DOCX.\n
Finding relevant information means sifting through multiple documents to locate concrete answers. Imagine being able to chat with these documents.\n

If you use ChatGPT or Gemini to upload your documents to chat with them, then that risks making your company's confidential information public.

RAGify offers a secure mechanism to:
- Build a secure system that works with your internal documents
- Integrate with trusted external entities
- Maintain the confidentiality of your data

RAGify offers a simple, secure way to bring your static documents to life using the power of Generative AI.

#### Navigation

Click on the sidebar to interact with RAGify App. \n
If you would like to look at the code of the app, then [jump to Github](https://github.com/kanad13/RAGify). \n
If you would like to know more about me, **Kunal Pathak**, the creator of this app, then [checkout my website](https://www.kunal-pathak.com).
				 """)
