import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="RAGify", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### RAGify - Chat with Your Documents using Gen AI

RAGify is a chatbot.

It is designed to search for information across many documents.

Simply upload your documents, and when you ask a question, it searches across them to find accurate answers.

RAGify makes your documents interactive using Generative AI and Retrieval-Augmented Generation (RAG).

_**Click on the sidebar to interact with RAGify App.**_

#### More details


Check out the code of the chatbot [on Github.](https://github.com/kanad13/RAGify)

Checkout my website for other AI/ML projects - [Kunal-Pathak.com](https://www.kunal-pathak.com).
				 """)
