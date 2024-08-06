import streamlit as st

default_questions = [
    "Select a question",
    "What's the weather like today?",
    "Tell me a joke",
    "What's the capital of France?",
    "How do I make pasta?",
    "Other (Type your own question)"
]

selected_question = st.selectbox("Choose a question:", default_questions)

if selected_question in ["Select a question", "Other (Type your own question)"]:
    user_question = st.text_input("Enter your question:")
else:
    user_question = selected_question
