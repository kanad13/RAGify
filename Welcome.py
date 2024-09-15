import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="RAGify", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### RAGify - Chat with Your Documents using Gen AI

This chatbot is for RAGify - a tool that lets you chat with your documents using Generative AI & Retrieval-Augmented Generation (RAG).

Think of RAGify as giving your documents a brain.

#### Why RAGify?

People and businesses often collect many documents in formats like PDF and DOCX.\n
To find specific information, they have to search through all these documents. Imagine if you could just chat with these documents to get the answers you need easily.\n

RAGify provides a secure way to:

- Create a private system for your documents
- Connect with trusted external sources
- Keep your data confidential

RAGify offers an easy and secure method to make your documents **interactive** using Generative AI.

#### Navigation

Click on the sidebar to interact with RAGify App. \n
If you would like to look at the code of the app, then [jump to Github](https://github.com/kanad13/RAGify). \n
If you would like to know more about me, **Kunal Pathak**, the creator of this app, then [checkout my website](https://www.kunal-pathak.com).
				 """)
