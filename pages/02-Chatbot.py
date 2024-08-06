import os
import groq
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import logging
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Groq client with the API key obtained from environment variables
client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    #api_key=st.secrets["GROQ_API_KEY"],
)

# Load a pre-trained sentence transformer model for generating embeddings
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

# Function to extract text content from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Function to split the extracted text into smaller chunks for processing
def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Process all PDF files in the specified directory and create chunks from their text
@st.cache_data
def process_pdfs():
    pdf_directory = './input_files/'
    all_chunks = []
    chunk_to_doc = {}
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = create_chunks(text)
            all_chunks.extend(chunks)
            for chunk in chunks:
                chunk_to_doc[chunk] = filename

    # Log information about chunks
    logging.info(f"Total chunks: {len(all_chunks)}")
    logging.info(f"Sample chunk: {all_chunks[0][:100]}...")  # Print first 100 characters of the first chunk

    return all_chunks, chunk_to_doc

# Generate embeddings and create FAISS index
@st.cache_resource
def create_faiss_index(all_chunks):
    embeddings = model.encode(all_chunks)
    dimension = embeddings.shape[1]
    num_chunks = len(all_chunks)

    # Dynamically decide on the index type based on the number of chunks
    if num_chunks < 100:
        logging.info("Using FlatL2 index due to small number of chunks")
        index = faiss.IndexFlatL2(dimension)
    else:
        logging.info("Using IVFFlat index")
        n_clusters = min(int(np.sqrt(num_chunks)), 100)  # Adjust number of clusters based on data size
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        index.train(embeddings.astype('float32'))

    index.add(embeddings.astype('float32'))
    return index

# Initialize cache
cache_file = 'semantic_cache.json'

# Function to load the cache from a JSON file
def load_cache():
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            if cache.get('model_name') != model_name:
                logging.info("Embedding model changed. Resetting cache.")
                return {"queries": [], "embeddings": [], "responses": [], "model_name": model_name}
            return cache
    except FileNotFoundError:
        return {"queries": [], "embeddings": [], "responses": [], "model_name": model_name}

# Function to save the cache to a JSON file
def save_cache(cache):
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

# Load the cache
cache = load_cache()

# Function to retrieve a response from the cache based on query similarity
def retrieve_from_cache(query_embedding, threshold=0.5):
    for i, cached_embedding in enumerate(cache['embeddings']):
        if len(cached_embedding) != len(query_embedding):
            logging.warning("Cached embedding dimension mismatch. Skipping cache entry.")
            continue
        distance = np.linalg.norm(query_embedding - np.array(cached_embedding))
        if distance < threshold:
            return cache['responses'][i]
    return None

# Function to update the cache with a new query, embedding, and response
def update_cache(query, query_embedding, response):
    cache['queries'].append(query)
    cache['embeddings'].append(query_embedding.tolist())
    cache['responses'].append(response)
    cache['model_name'] = model_name
    save_cache(cache)

# Function to retrieve the most relevant chunks of text based on a query
def retrieve_relevant_chunks(query, index, all_chunks, top_k=10):
    query_vector = model.encode([query])[0]

    cached_response = retrieve_from_cache(query_vector)
    if cached_response:
        logging.info("Answer recovered from Cache.")
        return cached_response

    top_k = min(top_k, len(all_chunks))  # Ensure we don't request more chunks than available
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    relevant_chunks = [all_chunks[i] for i in I[0]]

    update_cache(query, query_vector, relevant_chunks)
    return relevant_chunks

# Function to generate a response using the Groq API based on relevant chunks
def generate_response(query: str, relevant_chunks: List[str], model: str = "llama-3.1-8b-instant"):
    context = "\n".join(relevant_chunks)
    prompt = f"""Based on the following context, please answer the question. If the answer is not fully contained in the context, provide the most relevant information available and indicate any uncertainty.

Context:
{context}

Question: {query}

Answer:"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the given context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=model,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None
    )

    response = chat_completion.choices[0].message.content.strip()
    usage_info = {
        "prompt_tokens": chat_completion.usage.prompt_tokens,
        "completion_tokens": chat_completion.usage.completion_tokens,
        "total_tokens": chat_completion.usage.total_tokens
    }
    logging.info(f"Usage Info: {usage_info}")
    return response, usage_info, relevant_chunks

# Function to process a query using retrieval-augmented generation (RAG)
def rag_query(query: str, index, all_chunks, chunk_to_doc, top_k: int = 10) -> tuple:
    relevant_chunks = retrieve_relevant_chunks(query, index, all_chunks, top_k)
    response, usage_info, used_chunks = generate_response(query, relevant_chunks)
    source_docs = list(set([chunk_to_doc[chunk] for chunk in used_chunks]))
    return response, usage_info, source_docs

# Streamlit app
st.set_page_config(page_title="Blunder Mifflin", page_icon=":soccer:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

def main():
    st.subheader("Ask questions about Blunder Mifflin's Company Policy and get AI-powered answers.")

    # Process PDFs and create index
    all_chunks, chunk_to_doc = process_pdfs()
    index = create_faiss_index(all_chunks)

    # Default questions
    default_questions = [
        "Select a question",
        "What is Blunder Mifflin's product range?",
        "Who is part of Blunder Mifflin's team?",
        "Describe Blunder Mifflin's remote work policy?",
        "How can one raise grievances in Blunder Mifflin?",
        "What is Blunder Mifflin's policy on smoking at office?",
        "What is Blunder Mifflin's policy relationships and nepotism?",
        "What is Blunder Mifflin's policy prank protocol?",
        "Describe Blunder Mifflin's Birthday Party Committee Rules",
        "Other (Type your own question)"
    ]

    # User input with default questions using dropdown
    selected_question = st.selectbox("Choose a question or select 'Other' to type your own:", default_questions)

    # Text input for custom questions or editing selected questions
    if selected_question in ["Select a question", "Other (Type your own question)"]:
        user_query = st.text_input("Enter your question:")
    else:
        user_query = st.text_input("Edit the selected question or enter a new one:", value=selected_question)

    if st.button("Get Answer"):
        if user_query and user_query != "Select a question":
            with st.spinner("Generating answer..."):
                response, usage_info, source_docs = rag_query(user_query, index, all_chunks, chunk_to_doc)

            # Display the response
            st.subheader("Answer:")
            st.write(response)

            # Display the source documents
            st.subheader("Source Documents:")
            for doc in source_docs:
                st.write(f"- {doc}")

            # Display usage information
            with st.expander("Usage Information"):
                st.json({
                    "Prompt Tokens": usage_info["prompt_tokens"],
                    "Completion Tokens": usage_info["completion_tokens"],
                    "Total Tokens": usage_info["total_tokens"]
                })
        else:
            st.warning("Please select a question or enter your own.")

if __name__ == "__main__":
    main()
