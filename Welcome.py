import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="RAGify", page_icon=":soccer:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

st.write("""
#### RAGify - Bringing Your Documentation to Life with Generative AI

Ever wanted to chat with your company's documents? RAGify makes it possible.

RAGify shows how to turn your internal knowledge base into an interactive Q&A system using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs). It's like giving your PDFs a brain.

While tools like ChatGPT or Gemini let you upload docs, using public models for company documents is risky. RAGify shows you how easy it is build a system that works on your internal documents without exposing your confidential corporate data to anyone outside your company.

#### Blunder Mifflin

To demonstrate how RAG works with LLM on custom documents, I created the Employee Handbook for a fictional compnay called Blunder Mifflin. The chatbot answers questions related to the company policy.\n
So for example, if an employee of Blunder Mifflin wants to know what is the "Work from Home" policy, then they can just ask the chatbot that question and get the answer using the power of Generative AI.
If the company's Work From Home policy gets updated, they just update the documents, no changes needed to the chatbot. The chatbot starts providing answers based on latest information.

#### Check out

Checkout how the bot works by clicking on the "Chatbot" button in the sidebar!\n
For more details, visit my [blog post](https://www.kunal-pathak.com/blog/RAGify).\n
Or dig into the code on  [Github](https://github.com/kanad13/RAGify).
				 """)

#st.image('late/projects/UEFA-Euro-Data-Analysis/assets/soccer_player.jpeg', caption=None, width=400, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
