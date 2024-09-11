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
import hashlib
import time
from groq import RateLimitError

# I have loaded environment variables to keep sensitive information out of the codebase.
# This is crucial for security and allows for easy configuration changes across environments.
load_dotenv()

# I have set up logging to track execution and debug issues.
# Proper logging is essential for monitoring and troubleshooting in production environments.
logging.basicConfig(level=logging.INFO)

# I have initialized the Groq client for API access.
# Here, I'm using an API key stored in environment variables for security.
# The commented out line shows an alternative using Streamlit secrets, which is useful for deployment scenarios.
client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    #api_key=st.secrets["GROQ_API_KEY"], # GROQ_API_KEY = ""
)

# I have loaded a pre-trained sentence transformer model for generating text embeddings.
# I chose 'all-mpnet-base-v2' for its balance of performance and accuracy.
# This model is crucial for converting text to vector representations for similarity search.
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

def extract_text_from_pdf(pdf_path):
    # I have extracted text from PDFs to make the content searchable.
    # This allows us to work with various document formats in a unified way.
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    # I have chunked the text for two main reasons:
    # 1. It allows us to process long documents that might exceed model token limits.
    # 2. It creates more granular pieces of text, improving retrieval accuracy.
    # I have used overlap to maintain context between chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_files_hash(directory):
    # I have hashed the input files to detect changes.
    # This is crucial for maintaining an up-to-date knowledge base without unnecessary reprocessing.
    hash_md5 = hashlib.md5()
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.pdf'):
            with open(os.path.join(directory, filename), "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
    return hash_md5.hexdigest()

@st.cache_data
def process_pdfs(_hash=None):
    # I have used caching to avoid reprocessing PDFs on every run.
    # This significantly improves performance for repeated queries.
    pdf_directory = './input_files/'
    current_hash = get_files_hash(pdf_directory)

    # I have cleared the cache if input files have changed.
    # This ensures we're always working with the most up-to-date information.
    if _hash is not None and _hash != current_hash:
        st.cache_data.clear()

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

    # I have used logging to help with debugging and monitoring the chunking process.
    logging.info(f"Total chunks: {len(all_chunks)}")
    logging.info(f"Sample chunk: {all_chunks[0][:100]}...")

    return all_chunks, chunk_to_doc, current_hash

@st.cache_resource
def create_faiss_index(all_chunks):
    # I have used FAISS for efficient similarity search.
    # This is crucial for quickly finding relevant chunks when answering queries.
    embeddings = model.encode(all_chunks)
    dimension = embeddings.shape[1]
    num_chunks = len(all_chunks)

    # I have dynamically chosen the index type based on the dataset size.
    # This optimizes search performance: FlatL2 for small datasets, IVFFlat for larger ones.
    if num_chunks < 100:
        logging.info("Using FlatL2 index due to small number of chunks")
        index = faiss.IndexFlatL2(dimension)
    else:
        logging.info("Using IVFFlat index")
        n_clusters = min(int(np.sqrt(num_chunks)), 100)  # Balancing clustering and search efficiency
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        index.train(embeddings.astype('float32'))

    index.add(embeddings.astype('float32'))
    return index

# I have initialized a cache for storing query results.
# Caching improves response times for repeated or similar queries.
cache_file = 'semantic_cache.json'

def load_cache():
    # I have loaded the cache from a file to persist it across sessions.
    # This improves the system's efficiency over time.
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            # I have reset the cache if the embedding model changes to ensure consistency.
            if cache.get('model_name') != model_name:
                logging.info("Embedding model changed. Resetting cache.")
                return {"queries": [], "embeddings": [], "responses": [], "model_name": model_name}
            return cache
    except FileNotFoundError:
        return {"queries": [], "embeddings": [], "responses": [], "model_name": model_name}

def save_cache(cache):
    # I have regularly saved the cache to ensure we don't lose valuable precomputed results.
    with open(cache_file, 'w') as f:
        json.dump(cache, f)

cache = load_cache()

def retrieve_from_cache(query_embedding, threshold=0.5):
    # I have implemented semantic caching to reuse results for similar queries.
    # This significantly reduces API calls and improves response times.
    for i, cached_embedding in enumerate(cache['embeddings']):
        if len(cached_embedding) != len(query_embedding):
            logging.warning("Cached embedding dimension mismatch. Skipping cache entry.")
            continue
        distance = np.linalg.norm(query_embedding - np.array(cached_embedding))
        if distance < threshold:
            return cache['responses'][i]
    return None

def update_cache(query, query_embedding, response):
    # I have updated the cache with new queries to continually improve performance.
    cache['queries'].append(query)
    cache['embeddings'].append(query_embedding.tolist())
    cache['responses'].append(response)
    cache['model_name'] = model_name
    save_cache(cache)

def retrieve_relevant_chunks(query, index, all_chunks, top_k=10):
    # I have used vector similarity to find the most relevant chunks.
    # This is more effective than keyword matching for understanding context and semantics.
    query_vector = model.encode([query])[0]

    cached_response = retrieve_from_cache(query_vector)
    if cached_response:
        logging.info("Answer recovered from Cache.")
        return cached_response

    # I have limited top_k to avoid retrieving more chunks than available.
    top_k = min(top_k, len(all_chunks))
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    relevant_chunks = [all_chunks[i] for i in I[0]]

    update_cache(query, query_vector, relevant_chunks)
    return relevant_chunks

def generate_response(query: str, relevant_chunks: List[str], primary_model: str = "llama-3.1-8b-instant", fallback_model: str = "gemma2-9b-it", max_retries: int = 3):
    # I have used a language model to generate responses based on retrieved chunks.
    # This allows for more natural and contextually appropriate answers.
    context = "\n".join(relevant_chunks)
    prompt = f"""Based on the following context, please answer the question. If the answer is not fully contained in the context, provide the most relevant information available and indicate any uncertainty.

Context:
{context}

Question: {query}

Answer:"""

    # I have implemented a fallback mechanism and retry logic for robustness.
    # This ensures the system can handle API errors and rate limits gracefully.
    models = [primary_model, fallback_model]
    for model in models:
        for attempt in range(max_retries):
            try:
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
                    "total_tokens": chat_completion.usage.total_tokens,
                    "model_used": model
                }
                logging.info(f"Usage Info: {usage_info}")
                return response, usage_info, relevant_chunks

            except RateLimitError as e:
                if model == fallback_model and attempt == max_retries - 1:
                    logging.error(f"Rate limit exceeded for both models after {max_retries} attempts.")
                    raise e
                logging.warning(f"Rate limit exceeded for model {model}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error occurred with model {model}: {str(e)}")
                break  # Move to the next model if any other error occurs

    raise Exception("Failed to generate response with all available models.")

def rag_query(query: str, index, all_chunks, chunk_to_doc, top_k: int = 10) -> tuple:
    # I have combined retrieval and generation for a complete RAG pipeline.
    # RAG allows us to ground the model's responses in specific, relevant information.
    relevant_chunks = retrieve_relevant_chunks(query, index, all_chunks, top_k)
    response, usage_info, used_chunks = generate_response(query, relevant_chunks)

    # I have tracked source documents for transparency and citation.
    source_docs = {}
    for chunk in used_chunks:
        doc_name = chunk_to_doc.get(chunk, "Unknown Source")
        if doc_name not in source_docs:
            source_docs[doc_name] = []
        source_docs[doc_name].append(chunk)

    return response, usage_info, source_docs

# I have configured the Streamlit app.
# I have used Streamlit for rapid prototyping and easy deployment of the user interface.
st.set_page_config(page_title="Blunder Mifflin", page_icon=":chat-plus-outline:", layout="wide", initial_sidebar_state="expanded", menu_items=None)

def main():
    st.write("Ask questions about Blunder Mifflin's Company Policy.")
    # I have processed PDFs and created the index at the start to ensure up-to-date information.
    all_chunks, chunk_to_doc, current_hash = process_pdfs()
    index = create_faiss_index(all_chunks)
    # I have provided default questions to guide users and demonstrate system capabilities.
    default_questions = [
        "Select a question",
        "What is Blunder Mifflin's product range?",
        "Who is part of Blunder Mifflin's team?",
        "What is Blunder Mifflin's policy relationships and nepotism?",
        "Describe Blunder Mifflin's Birthday Party Committee Rules",
        "Other (Type your own question)"
    ]
    # I have used a dropdown for ease of use, but also allowed custom questions for flexibility.
    selected_question = st.selectbox("Choose a question or select 'Other' to type your own:", default_questions)
    if selected_question == "Other (Type your own question)":
        user_query = st.text_input("Enter your question:")
    elif selected_question != "Select a question":
        user_query = selected_question
    else:
        user_query = ""
    if user_query:
        pass
    # I have used a button to trigger the query process, giving users control over when to send a request.
    if st.button("Get Answer"):
        if user_query and user_query != "Select a question":
            with st.spinner("Generating answer..."):
                # I have rechecked for changes in PDFs to ensure we're using the latest data.
                all_chunks, chunk_to_doc, _ = process_pdfs(current_hash)
                index = create_faiss_index(all_chunks)
                response, usage_info, source_docs = rag_query(user_query, index, all_chunks, chunk_to_doc)
            # I have displayed the response, sources, and usage info for transparency.
            st.subheader("Answer:")
            st.write(response)
            st.subheader("Source Documents:")
            for doc_name, chunks in source_docs.items():
                with st.expander(f"Text used from {doc_name}"):
                    for chunk in chunks:
                        st.markdown(chunk)

            # Track tokens used by each query
            st.subheader("Track LLM Usage")
            st.json({
                "Prompt Tokens": usage_info["prompt_tokens"],
                "Completion Tokens": usage_info["completion_tokens"],
                "Total Tokens": usage_info["total_tokens"],
                "Model Used": usage_info["model_used"]
            })
        else:
            st.warning("Please select a question or enter your own.")

if __name__ == "__main__":
    main()
