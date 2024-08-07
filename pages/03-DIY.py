import streamlit as st

config = {'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False}
st.set_page_config(page_title="DIY", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)
st.write("""
### My motivation

My goal with RAGify is to demystify this type of AI application.\n
Whether you are a developer looking to experiment or a business leader trying to understand how such a system works, RAGify shows you how accessible this technology really is.\n
[Check out the code for RAGify here.](https://github.com/kanad13/RAGify)\n

### Simple Explanations

If some of the terms used here are new to you, then you can check out the [simple explanations and diagrams I have made here.](https://github.com/kanad13/RAGify#simple-explanations-for-complex-terms)\n

### Do It Yourself (DIY)

Would you like to use RAGify with your own documents?\n
Of course you can with a few simple tweaks. That is the best part - that you can build this too!

RAGify is not using any groundbreaking new tech - it is just combining existing tools in a useful way.\n
I have made all the code available, along with this working demo, so you can see exactly how it is done.\n
[Check out these simple steps to make RAGify your own.](https://github.com/kanad13/RAGify#rag-with-your-own-documents)
""")
