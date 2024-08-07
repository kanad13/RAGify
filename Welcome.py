import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="RAGify", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### RAGify - Bringing Your Documentation to Life with Generative AI

This chatbot is for RAGify - a tool that shows you how to transform your internal company documents into an interactive Q&A system.\n

It uses two key technologies:

1. Retrieval-Augmented Generation (RAG)
2. Large Language Models (LLMs)

#### Why RAGify?

Every company has a vast amount of documentation spread across various file formats like PDF and DOCX.\n
Would it not be great if you could chat with these documents?

If you use ChatGPT or Gemini to upload your documents to chat with them, then that risks making your company's confidential information public.

RAGify provides an alternative by showing you how to:
- Build a secure system that works with your internal documents
- Integrate with trusted external entities
- Maintain the confidentiality of your corporate data

RAGify offers a simple, secure way to bring your static documents to life using the power of Generative AI.

#### Navigation

Click on the sidebar to interact with RAGify App. \n
If you would like to look at the code of the app, then [jump to Github](https://github.com/kanad13/RAGify). \n
If you would like to know more about me, **Kunal Pathak**, the creator of this app, then [checkout my website](https://www.kunal-pathak.com).
				 """)
