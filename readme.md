# RAGify - Turn Your Documents into an Interactive AI Assistant

Ever wanted to chat with your company's documents? RAGify makes it possible.

This github repository is for RAGify - a tool that transforms your internal documents into an interactive Q&A system. It uses two key technologies:

1. Retrieval-Augmented Generation (RAG)
2. Large Language Models (LLMs)

Think of RAGify as giving your PDFs a brain, without compromising your confidential data.

[Click here to see RAGify in action on some sample documents.](https://ragify.streamlit.app)

## Why RAGify?

Every company has a vast amount of documentation spread across various file formats like PDF and DOCX. Wouldn't it be great if you could chat with these documents?
If you use ChatGPT or Gemini to upload yourn documents to chat with them, then that risks making your company's confidential information public.

RAGify provides an alternative by showing you how to:

- Build a secure system that works with your internal documents
- Integrate with trusted external entities
- Maintain the confidentiality of your corporate data

RAGify offers a simple, secure way to bring your static documents to life using the power of Generative AI.

## Customize RAGify for Your Needs

This Github repository provides detailed documentation on how RAGify works.

With this information, you can quickly adapt RAGify to suit your needs.

By making a few changes, RAGify can work with your own documentation.

Scroll below to see how you can build a similar system.

## How RAGify Works?

```mermaid
graph TD
    subgraph User
    A[Input Query]
    H[Get Answer]
    end
    subgraph Knowledge Base
    I[Document Storage]
    J[Vector Embeddings]
    end
    subgraph RAG System
    B[Embed Query]
    C[Vector Search]
    D[Retrieve Relevant Texts]
    E[Create Prompt]
    end
    subgraph LLM
    F[Process Prompt]
    G[Generate Response]
    end
    A --> B
    I --> J
    J --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

Here is a simple overview of the diagram above:

1. The user asks a question.
2. The RAG system converts this question into a numerical format (vector) that computers can understand and compare easily.
3. It then searches through a database of pre-converted document vectors to find the most relevant information.
4. The system retrieves the actual text of these relevant documents.
5. It combines the user's question with this relevant information to create a detailed prompt.
6. This prompt is sent to an AI (the LLM), which processes it and generates a response.
7. Finally, the user receives this response as their answer.

## What is covered in this repository?

1. [RAGify - Turn Your Documents into an Interactive AI Assistant](#ragify---turn-your-documents-into-an-interactive-ai-assistant)
   1. [Why RAGify?](#why-ragify)
   2. [Customize RAGify for Your Needs](#customize-ragify-for-your-needs)
   3. [How RAGify Works?](#how-ragify-works)
   4. [What is covered in this repository?](#what-is-covered-in-this-repository)
   5. [Key Components of RAGify](#key-components-of-ragify)
   6. [Simple Explanations for Complex Terms](#simple-explanations-for-complex-terms)
   7. [Input Documents - Blunder Mifflin](#input-documents---blunder-mifflin)
   8. [RAG with your own documents](#rag-with-your-own-documents)
   9. [Technical Deepdive into RAGify](#technical-deepdive-into-ragify)
   10. [Acknowledgements](#acknowledgements)

## Key Components of RAGify

- **Large Language Model (LLM) to generate context-aware responses.**
  - My project uses Groq's API to interact with the large language model, specifically 8B model from the [Meta Llama 3.1 collection](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md).
  - This model is served by [Groq Cloud](https://wow.groq.com/why-groq/) and is responsible for producing intelligent and contextually relevant responses based on the retrieved chunks of text from the input documents.
  - The code can be easily customized to use other models like `OpenAI`, `Mistral`, or `Gemini`.
- **Sentence Transformers to generate embeddings for text chunks and queries.**
  - The `SentenceTransformer` model [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) is used to create dense vector embeddings from the text data extracted from PDFs.
  - These embeddings are crucial for semantic search, allowing the system to find the most relevant text chunks corresponding to a user's query.
- **FAISS (Facebook AI Similarity Search) to perform similarity search over large datasets**
  - FAISS is employed to perform fast and accurate nearest neighbor searches among the generated embeddings.
  - Depending on the number of chunks, different FAISS index types (`IndexFlatL2` for small datasets or `IndexIVFFlat` for larger datasets) are dynamically selected to balance speed and accuracy.
- **PyPDF for pdf extraction**
  - Usage - The `PdfReader` from the [pypdf library](https://pypdf.readthedocs.io/en/latest/index.html) is used to read and extract text content from PDF files, which is then processed and chunked for further analysis.
  - This allows the system to work with document-based data sources.
- **Langchain for text chunking and splitting.**
  - The `RecursiveCharacterTextSplitter` from [Langchain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) is used to break down large chunks of text into smaller, more manageable pieces.
  - This facilitates efficient processing and ensures that the text chunks are of an appropriate size for embedding and retrieval.
- **Caching Mechanism to speed up query responses by reusing previous results.**
  - The system includes a JSON-based caching mechanism to store previous queries, their embeddings, and responses.
  - Before generating a new response, the system checks the cache to see if a similar query has already been processed.
  - If found, the cached response is returned, significantly improving response time and reducing the need for redundant computations.
- **Streamlit for creating a web-based user interface.**
  - Streamlit is used to create an interactive web application that allows users to input queries and view responses generated by the system.
  - It provides an intuitive and easy-to-deploy interface for users to interact with the RAG system without needing to run the code manually.

## Simple Explanations for Complex Terms

This section explains some of the complex terms used in this project.

1. [Prompt Engineering](./readme.md#prompt-engineering)
2. [Fine-tuning](./readme.md#fine-tuning)
3. [Retrieval-Augmented Generation (RAG)](./readme.md#retrieval-augmented-generation-rag)
4. [Vector Database](./readme.md#vector-database)
5. [Retriever](./readme.md#retriever)

### Prompt Engineering

The questions you ask a Generative AI model (e.g., ChatGPT, Gemini) are called "prompts." To get better answers, you must "engineer" or "refine" your prompt.

Example - Instead of saying "Write about dogs," say "Write a 200-word paragraph about the history of domesticated dogs, focusing on their roles in human society."

```mermaid
graph LR
    A[User Input] --> B[Simple Prompting]
    A --> C[Prompt Engineering]

    B --> D[Direct Output]

    C --> E{Refine Prompt}
    E -->|Improve| F[Better Results]
    E -->|Iterate| C
    F --> G[Final Output]

    subgraph Simple Prompting
    B
    D
    end

    subgraph Prompt Engineering
    C
    E
    F
    end
```

### Fine-tuning

Fine-tuning makes an AI model better at specific tasks. It is like teaching a smart student to become an expert in a new subject.

Let us say a law firm needs to create many legal documents every day. ChatGPT can write these documents, but it might not use the right words or the right format, that the law firm needs.

To fix this, the law firm can "fine-tune" the AI model. This means teaching the AI to write documents exactly how the law firm wants them.

To do this, the firm shows the AI model some examples of perfect legal documents written by their best lawyers. The AI learns from these examples and gets better at writing documents just like the law firm wants.

```mermaid
flowchart LR
    A[("Base LLM")] -->|Fine-tuning| B
    subgraph FT [Fine-tuned LLM]
        B[("Fine-tuned Model")]
        D[("Fine-tuned Knowledge")]
    end
    C[("Law-firm specific examples")] --> FT
    FT <-->|Prompt/Response| E[("User")]
```

### Retrieval-Augmented Generation (RAG)

RAG allows you to feed the AI model your own data sources, enabling it to give more relevant and tailored responses.

Imagine a pizza restaurant's chatbot using RAG. It is like giving the chatbot a constantly updated menu card. When customers ask about today's specials, changed delivery zones, or new toppings, the chatbot can instantly access this fresh information. It does not just rely on old data but can pull up the latest details.

```mermaid
graph TB
    subgraph "Traditional AI Model"
    A1[User Query] --> B1[AI Model]
    B1 --> C1[General Response]
    end
    subgraph "RAG-Enabled AI Model"
    A2[User Query] --> R[Retriever]
    R <--> V[Vector DB]
    V <--> D[Custom Data Sources]
    R --> B2[AI Model]
    A2 --> B2
    B2 --> C2[Tailored Response]
    end
```

### Vector Database

A Vector Database is a smart storage system for AI. It helps AI access new or specific information not included in its original training.

A vector database is like a smart library where instead of searching for books by their title or author, you are searching by the ideas inside the books. This helps the AI find and compare information more effectively.

Let me reuse the example from the previous section about the chatbot for a pizza restaurant. The restaurant keeps its latest menu in a Vector Database. When customers ask about new pizzas, the chatbot can quickly check this database for current information. This way, the restaurant does not need to constantly update the chatbot. They just add new pizza details to the database, and the chatbot can access this information when needed.

Vector databases store data differently from relational databases like MySQL. Instead of using rows and columns, vector databases convert each piece of data into a numerical format called an embedding. These embeddings are placed in a multi-dimensional space. Similar items are positioned closer together.

For example, "cats" and "dogs" would be near each other, while "table" and "chair" would be further apart. This method helps AI models provide more relevant answers.

![](./assets/vector_db_emeddings.svg)

### Retriever

The Retriever in a RAG system works like a smart search tool. It helps connect what users ask with the information stored in "vector databases".

When someone asks a question, the Retriever does three main things:

1. **Find Similar Info**: It looks for information that is close to what the user asked.

2. **Sort by Importance**: It puts the found information in order, with the most useful stuff at the top.

3. **Pick the Best**: It chooses the top pieces of information to send back to the AI.

The AI then uses this information to give an answer the user can easily understand.

```mermaid
graph TB
    subgraph "RAG-Enabled AI Model"
    A[User Query] --> R[Retriever]
    R <--> V[Vector DB]
    V <--> D[Custom Data Sources]
    R --> B[AI Model]
    A --> B
    B --> C[Tailored Response]
    end
```

## Input Documents - Blunder Mifflin

Now that I have explained some complex terms about Retrieval-Augmented Generation, let me get back to RAGify - the project to understand how you can use a chatbot on your own documents.

To showcase how RAGify works with LLM on custom documents, I created the Employee Handbook for a fictional company called Blunder Mifflin.

The chatbot answers questions related to the company policy.\n

So for example, if an employee of Blunder Mifflin wants to know what is the "Work from Home" policy, then they can just ask the chatbot that question and get the answer using the power of Generative AI.

If the company's Work From Home policy gets updated, they just update the documents, no changes needed to the chatbot. The chatbot starts providing answers based on latest information.

This approach allows me to explore the functionality of RAGify without using real, sensitive data.

You can check out the contents of the documents here - [Blunder Mifflin Handbook](./Blunder_Mifflin-Employee_Handbook.md).

The 5 PDF documents include:

1. Blunder Mifflin's History and Mission
2. Blunder Mifflin's Employee Handbook
3. In-Office Relationships Policy
4. Prank Protocol
5. Birthday Party Committee Rules

In the next few sections, I will use these 5 documents as sample input to demonstrate how RAGify can turn static information into an interactive, AI-driven Q&A system. By feeding these PDFs into the RAGify system, you will be able to ask questions like:

- "What is Blunder Mifflin's mission?"
- "What's the policy on in-office relationships?"
- "How should pranks be conducted at Blunder Mifflin?"

RAGify will process these documents, breaking them down into manageable chunks, and store them for quick retrieval.
When you ask a question, RAGify searches through these chunks to find the most relevant information, then uses AI to craft a clear and concise response.

This code will show how you can use RAG to handle your own company's documents.

## RAG with your own documents

In just a few steps, you can set up the RAG system for your own documents. Follow the instructions below:

### Install Prerequisites

- **Python** - Install Python by following [this guide](https://wiki.python.org/moin/BeginnersGuide/Download).
- **Git** - Install Git by following [these instructions](https://docs.github.com/en/get-started/getting-started-with-git/set-up-git).

### Clone the Repository

Clone the RAGify repository to your local machine:

```bash
git clone https://github.com/kanad13/RAGify.git
cd RAGify
```

### Create a Virtual Environment

Create a virtual environment to manage dependencies:

```bash
python -m venv rag_venv
```

Activate the virtual environment:

- On Windows:

```sh
.\rag_venv\Scripts\activate
```

- On Mac:

```sh
source rag_venv/bin/activate
```

### Install Required Packages

Install all the necessary Python packages:

```bash
pip install -r requirements.txt
```

### Set Up LLM

RAGify utilizes the open-source Meta Llama model, hosted by Groq. To use it, you'll need an API key. Follow [these instructions](https://console.groq.com/docs/quickstart) to obtain your key.

If you prefer using a different LLM, you can obtain API keys from the following providers:

- [OpenAI](https://platform.openai.com/docs/quickstart)
- [Gemini](https://ai.google.dev/gemini-api/docs)
- [Anthropic](https://docs.anthropic.com/en/docs/quickstart)
- [Mistral](https://docs.mistral.ai/api/)

Alternatively, you can set up an LLM on your own machine using:

- [GPT4All](https://github.com/nomic-ai/gpt4all)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp)
- [LocalAI](https://github.com/mudler/LocalAI)

### Configure API Key

Once you have your API key, create a `.env` file in the root directory of the cloned repository. Add the following line to the `.env` file:

```text
GROQ_API_KEY="your_key"
```

This key will be loaded at runtime into the RAGify system, keeping it private and secure on your machine. The [python-dotenv package](https://pypi.org/project/python-dotenv/) handles this process.

### Prepare Your Input Files

Place all your documents inside the `input_files` folder. You may remove the existing files if needed.

### Access the RAGify Application

You are now ready to interact with your documents. You have two options:

- **Option 1**: Run the code in a Jupyter notebook if you're technically inclined. See the notebook [here](./RAGify-full_code.ipynb).
- **Option 2**: Access the system via a Streamlit web app. Open a terminal, navigate to the root of the repository, and run the following command:

```bash
streamlit run Welcome.py
```

## Technical Deepdive into RAGify

This section will look at the technical implementation details of the RAGify project.
If you are eager to browse through the code yourself, jump right [here](./RAGify-full_code.ipynb).

### Visualize Full RAGify Architecture

Before diving into the details of building RAGify, here is a high-level overview of the system:

```mermaid
flowchart TD
    A[Start] --> B[Load Environment Variables]
    B --> C[Initialize Groq Client]
    C --> D[Load PDFs]
    D --> E[Extract Text]
    E --> F[Create Chunks]
    F --> G[Generate Embeddings]
    G --> H{Dataset Size}
    H -- Small --> I[Create FlatL2 Index]
    H -- Large --> J[Create IVFFlat Index]
    I --> K[FAISS Index]
    J --> K
    L[User Query] --> M[Encode Query]
    M --> N{Check Cache}
    N -- Hit --> O[Return Cached Response]
    N -- Miss --> P[Search Index]
    K --> P
    P --> Q[Retrieve Relevant Chunks]
    Q --> R[Generate LLM Response]
    R --> S{API Error?}
    S -- Yes --> T[Retry or Use Fallback Model]
    T --> R
    S -- No --> U[Update Cache]
    U --> V[Display Results]
    O --> V

    subgraph "Initialization"
        B
        C
    end

    subgraph "Data Processing and Indexing"
        D
        E
        F
        G
        H
        I
        J
        K
    end

    subgraph "Query Processing and Response Generation"
        L
        M
        N
        O
        P
        Q
        R
        S
        T
        U
    end

    subgraph "User Interface"
        V
    end
```

This diagram might look complex, but the RAG concept is simple. It involves processing documents, creating embeddings, storing them in a vector database, and then using AI to generate responses based on user queries.

### Setting Up RAGify Environment

First, I import the necessary libraries and set up logging:

```python
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

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize the Groq client with the API key obtained from environment variables
client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load a pre-trained sentence transformer model for generating embeddings
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)
```

### Document Processing

#### Process Input PDF Files

Now that I have set up the development environment, my next priority is to process the PDF files that make up the [Blunder Mifflin Employee Handbook](./Blunder_Mifflin-Employee_Handbook.md).

For this step, I need to extract text content from the input PDF documents. This process transforms the unstructured data in PDFs into a format that can be easily processed and analyzed by AI.
I have placed the files inside the [input_files](./input_files/) folder.

To process these PDFs, I am using [PyPDF](https://pypdf.readthedocs.io/en/latest/index.html). It is a free and open-source Python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files.

here is the code I have written to extract text from PDFs:

```python
import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Process all PDF files in the specified directory
pdf_directory = './input_files/'
all_chunks = []
for filename in os.listdir(pdf_directory):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, filename)
        text = extract_text_from_pdf(pdf_path)
        chunks = create_chunks(text)
        all_chunks.extend(chunks)
```

This code converts the PDF documents into plain text, preparing them for the next stages, such as text chunking and embedding creation.

#### Text Chunking with LangChain

After extracting text from Blunder Mifflin's PDF documents, I will break it down into smaller, manageable chunks. This process, known as **text chunking**, is essential for several reasons:

1. It allows for more precise information retrieval.
2. It helps maintain context within each chunk.
3. It optimizes the input size for our embedding model.

I will use LangChain's `RecursiveCharacterTextSplitter` for this step, which splits the input text while maintaining semantic coherence within each chunk.

[LangChain](https://langchain.com) is a framework designed to facilitate building applications that combine large language models (like Gemini or GPT) with other tools and data sources.
It enhances basic Large Language Models (LLMs) by connecting different components (AI, web, databases, memory, etc.) in a structured way.

For example, LangChain can be used to create a chatbot that not only answers customer queries using a language model but also accesses customer information from a database, checks order status, and remembers past interactions for more personalized support.

`RecursiveCharacterTextSplitter` is a LangChain tool that breaks down large text into smaller, more manageable chunks. This is crucial because large language models perform better with shorter text inputs. It attempts to split text at logical points, such as sentences or paragraphs, to ensure the smaller sections remain coherent.

here is how I chunk the text using RecursiveCharacterTextSplitter:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_chunks(text, chunk_size=1000, chunk_overlap=200):
    """
    Split a large text into smaller chunks.

    Args:
    text (str): The input text to be split.
    chunk_size (int): The maximum size of each chunk.
    chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
    list: A list of text chunks.
    """
    # Create a RecursiveCharacterTextSplitter object with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    # Split the input text into chunks
    chunks = text_splitter.split_text(text)
    return chunks

# Create chunks for all documents
all_chunks = []
```

This chunking process is vital for RAGify's performance. By breaking down Blunder Mifflin documents into smaller, overlapping pieces, I ensure that:

1. My vector database can efficiently store and retrieve relevant information.
2. I maintain local context within each chunk, improving the quality of our embeddings.
3. I optimize the input size for our embedding model, which often has token limits.

The `chunk_overlap` parameter is particularly important as it helps maintain continuity between chunks. This overlap ensures that I do not lose context at the boundaries between chunks, which could happen if I made hard cuts in the middle of sentences or paragraphs.

With our documents now split into manageable chunks, I am ready for the next step in our RAG pipeline: creating embeddings for these chunks and storing them in our vector database.

### Embedding Creation with Sentence Transformers

After chunking our documents, the next step is to create embeddings for each chunk.
[Embeddings](./readme.md#vector-database) are numerical representations of data (like text, images, or other objects) in the form of vectors (lists of numbers).
These vectors capture the meaning or features of the data, allowing databases to compare, search, and organize similar items effectively.

For RAGify, I will use the `sentence-transformers` library, which provides pre-trained models specifically designed for generating high-quality text embeddings.

The [sentence-transformers library](https://sbert.net) is a Python package that facilitates the use of transformer models (like BERT, RoBERTa, and others) for sentence and text embedding tasks.
It builds on top of the [Hugging Face Transformers library](https://huggingface.co/docs/transformers/en/index) and simplifies the process of generating dense vector representations (embeddings) for sentences, paragraphs, or entire documents.

I have previously used sentence-transformers in two of my other projects:

- [Movie-Recommendation-Bot](https://github.com/kanad13/Movie-Recommendation-Bot)
- [Emotion_Detection_App](https://github.com/kanad13/Emotion_Detection_App)

For this project, I have chosen to use the [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model to map sentences and paragraphs to a 768-dimensional dense vector space. This model is known for its high performance in various semantic similarity tasks.

here is the Python code I have written to create embeddings using the sentence-transformers library:

```python
from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model for generating embeddings
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name)

def create_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks.

    Args:
        chunks (list): A list of text chunks to embed.

    Returns:
        list: A list of embedding vectors.
    """

    # Generate embeddings for all chunks
    embeddings = model.encode(chunks)

    return embeddings

# Create embeddings for all chunks
embeddings = create_embeddings(all_chunks)
```

In my implementation, each chunk is transformed into a fixed-size vector. Using the same model for both indexing and querying ensures consistent representations.

The embeddings created by this code serve as the foundation for our semantic search capability.
By converting text into dense vector representations, I can compare the similarity between different pieces of text.
This allows RAGify to quickly identify the most relevant chunks when responding to user queries, enhancing the overall performance and accuracy of the system.
In the next section, I will store these vectors in our FAISS database for fast and efficient similarity search.

### Vector Database Setup with FAISS

FAISS (Facebook AI Similarity Search) is a powerful library developed by Facebook Research for efficient similarity search and clustering of dense vectors.

[FAISS](https://github.com/facebookresearch/faiss) is particularly well-suited for RAGify because:

1. It is optimized for speed, allowing for fast similarity searches even with large datasets.
2. It supports various indexing algorithms, enabling us to balance between search speed and accuracy.
3. It can handle high-dimensional vectors, which is perfect for our embedding vectors.
4. It provides both CPU and GPU implementations for flexibility in different computing environments.

The FAISS index is selected based on the number of chunks, which dynamically adjusts the index type to optimize performance.

```python
import faiss
import numpy as np

# Create a FAISS index for efficient similarity search based on embeddings
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
```

In this implementation:

1. I dynamically decide whether to use the `IndexFlatL2` index for smaller datasets (fewer than 100 chunks) or the `IndexIVFFlat` index for larger datasets. The `IndexIVFFlat` index is more scalable and performs well on larger datasets by clustering the vectors.
2. For `IndexIVFFlat`, I adjust the number of clusters based on the size of the dataset, ensuring the index remains efficient even as the number of chunks increases.
3. I add our embeddings to the index using the `add` method.

By setting up this FAISS index, I have created a tool for quickly finding the most similar document chunks to any given query. This will be crucial in the next steps of our RAG pipeline, where I will use this index to retrieve relevant information in response to user queries.

### Caching Strategy for Performance Optimization

To further enhance the performance of RAGify, I have implemented a caching strategy. This allows the system to store and retrieve previously computed results, reducing the need for repetitive computations and improving response times for frequently asked questions.

- **How the Cache Works**

1. Cache Initialization

   - A cache is initialized and loaded from a JSON file (`semantic_cache.json`). This file stores previous queries, their corresponding embeddings, and the generated responses. The cache also tracks the model used to ensure consistency.

2. Cache Lookup

   - When a new query is received, the system first checks the cache to see if a similar query has been previously processed. The similarity is determined based on the distance between the new query's embedding and those stored in the cache.
   - If a match is found (within a defined threshold), the cached response is returned immediately, bypassing the need to recompute embeddings or search the vector database.

3. Cache Update

   - If the query is not found in the cache, the system processes it as usual—creating embeddings, retrieving relevant chunks, and generating a response.
   - The new query, its embedding, and the generated response are then added to the cache for future use.

4. Cache Persistence

   - The cache is periodically saved to the disk, ensuring that the data persists across different sessions.

here is how the cache is implemented:

```python
import json
import logging

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
```

By integrating this caching mechanism, RAGify can efficiently handle repeated queries, reducing the computational load and delivering faster responses to the user.

### RAG - Retrieval Workflow

This step involves retrieving relevant chunks of information based on a user's query and using them to generate an informed response.
The core of our RAG system is the ability to find the most relevant information from our document collection. I have implemented this using the following function:

```python
def retrieve_relevant_chunks(query, top_k=10):
    query_vector = model.encode([query])[0]

    cached_response = retrieve_from_cache(query_vector)
    if cached_response:
        logging.info("Answer recovered from Cache.")
        return cached_response

    top_k = min(top_k, len(all_chunks))  # Ensure we do not request more chunks than available
    D, I = index.search(np.array([query_vector]).astype('float32'), top_k)
    relevant_chunks = [all_chunks[i] for i in I[0]]

    update_cache(query, query_vector, relevant_chunks)
    return relevant_chunks
```

This function performs the following steps:

1. Encodes the user's query into a vector using the same SentenceTransformer model I used for our documents.
2. Searches the FAISS index for the `top_k` most similar vectors to the query vector.
3. Returns the corresponding text chunks from our original document set.
4. If the query is found in the cache, the system returns the cached response instead of performing the search again.

The `top_k` parameter allows me to control how many relevant chunks I retrieve, balancing between comprehensive context and computational efficiency.

### Visualize Caching aspect of RAGify

This section is kind of a summary of the topics handled till now.
First look at the diagram below

```mermaid
sequenceDiagram
participant PDF as PDF Files
participant PyPdf as PyPdf
participant LangChain as LangChain
participant SentenceTransformer as SentenceTransformer
participant FAISS as FAISS
participant Cache as Cache
participant Disk as Disk Storage

PDF->>PyPdf: Input
PyPdf->>LangChain: Extracted Text
LangChain->>SentenceTransformer: Text Chunks
SentenceTransformer->>FAISS: Embeddings
FAISS->>Cache: Cached Embeddings
Cache->>FAISS: Retrieve Relevant Chunks
FAISS->>Disk: Vector Index

Note over PDF,PyPdf: Text Extraction
Note over PyPdf,LangChain: Semantic Segmentation
Note over LangChain,SentenceTransformer: Vectorization
Note over SentenceTransformer,FAISS: Similarity Search Index Creation
Note over FAISS,Cache: Caching for Performance
Note over FAISS,Disk: Persistent Storage
```

This diagram shows the flow of my RAG system:

1. **Text Extraction** - PDF files are processed by PyPdf to extract raw text.
2. **Semantic Segmentation** - LangChain breaks down the extracted text into meaningful chunks.
3. **Vectorization** - SentenceTransformer converts text chunks into numerical embeddings.
4. **Similarity Search Index Creation** - FAISS organizes these embeddings for efficient retrieval.
5. **Caching for Performance** - Similar queries and responses are cached to improve response time.
6. **Persistent Storage** - The FAISS index is saved to disk for future use.

When a query is received, the system uses this prepared index to quickly retrieve relevant information, which is then used to augment the LLM's response. The next section will cover the LLM part.

### Response Generation using LLM

After setting up our vector database and implementing the retrieval of relevant chunks, the final step in our RAG system is to generate informed responses based on the retrieved information.

A Large Language Model generates contextual responses to your questions. For this project, I used the 8B model from the [Meta Llama 3.1 collection](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md), served by [Groq Cloud](https://wow.groq.com/why-groq/). However, the code can be easily customized to use other models like [OpenAI](https://platform.openai.com/docs/guides/chat-completions) or [Gemini](https://ai.google.dev/gemini-api/docs).

```python
import os
import groq
from typing import List

client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_response(query: str, relevant_chunks: List[str], model: str = "llama-3.1-8b-instant") -> str:
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
    usage_info = chat_completion.usage
    logging.info(f"Usage Info: {usage_info}")
    return response, usage_info
```

This function performs the following steps:

1. **Prompt Preparation** - It combines the retrieved relevant chunks into a single context string and formats a prompt that includes this context along with the user's query.
2. **API Call** - It uses Groq's API to generate a response. The `llama-3.1-8b-instant` model is used by default, but this can be customized.
3. **Response Extraction** - The generated response is extracted from the API's return value and cleaned up.

- **Define RAG Query**

And then this is the cherry on top - this function kind of ties it all together:

```python
def rag_query(query: str, top_k: int = 10) -> str:
    relevant_chunks = retrieve_relevant_chunks(query, top_k)
    response, usage_info = generate_response(query, relevant_chunks)
    # Uncomment the following line to log usage information
    # print(f"Usage Info: {usage_info}")
    return response
```

1. It retrieves the most relevant chunks using our previously implemented `retrieve_relevant_chunks` function.
2. It then passes these chunks along with the original query to the `generate_response` function.
3. Finally, it returns the generated response.

- **Testing the system locally**

To verify that our RAG system is working as expected, I use a test query:

```python
test_query = "Describe Blunder Mifflin's remote work policy?"
result = rag_query(test_query)
print(f"Query: {test_query}")
print(f"Response: {result}")
```

You will get an answer based on text from the Blunder Mifflin Employee Handbook.

### Streamlit App

Finally, to make RAGify accessible to non-technical users, I have built a user-friendly interface using Streamlit.

This section outlines how to set up and use the Streamlit app for querying your documents.

```python
def main():
    st.title("RAGify - Chat with Your Documents")
    st.write("Ask questions about your company's documents and get AI-powered answers.")

    # Process PDFs and create index
    # This calls our previously defined functions to process the PDFs and create the FAISS index. Note that these functions are decorated with StreamlIt is caching decorators (`@st.cache_data` and `@st.cache_resource`) to optimize performance.
    all_chunks, chunk_to_doc = process_pdfs()
    index = create_faiss_index(all_chunks)

    # User input
    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if user_query:
            with st.spinner("Generating answer..."):
                response, usage_info, source_docs = rag_query(user_query, index, all_chunks, chunk_to_doc)
      # When the user clicks the "Get Answer" button, this code checks if a query has been entered. If so, it calls our `rag_query` function to process the query. The `st.spinner` provides visual feedback while the answer is being generated.

            # Display the response
            st.subheader("Answer:")
            st.write(response)

            # Display the source documents
            st.subheader("Source Documents:")
            for doc in source_docs:
                st.write(f"- {doc}")
            # This section displays the generated answer, lists the source documents used to create the answer, and provides detailed usage information in an expandable section.

            # Display usage information
            with st.expander("Usage Information"):
                st.json(usage_info)
        else:
            st.warning("Please enter a question.")
```

### Visualize user engagement of RAGify

This diagram demonstrates how the end-user flow looks like.

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit UI
    participant R as RAG System
    participant F as FAISS Index
    participant G as Gemini LLM

    U->>S: Enter Query
    S->>R: Process Query
    R->>F: Retrieve Relevant Chunks
    F-->>R: Return Chunks
    R->>G: Generate Answer
    G-->>R: Return Answer
    R->>S: Display Results
    S->>U: Show Answer and Sources
```

## Acknowledgements

RAGify is powered by a combination of open and closed-source technologies. I am grateful for the contributions of the following initiatives and organizations:

- [Python](https://github.com/python) - The backbone of RAGify's codebase.
- [PyPDF](https://pypdf.readthedocs.io/en/latest/index.html) - It is used for text extraction and processing from PDF documents.
- [Hugging Face](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) - The `all-mpnet-base-v2` SentenceTransformer model is used to generate embeddings for semantic search.
- [Facebook](https://faiss.ai) - Facebook AI Similarity Search (FAISS) is used for performing similarity searches.
- [Langchain](https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/) - This RecursiveCharacterTextSplitter is used for breaking down large text into manageable chunks, optimizing them for embedding and retrieval.
- [Meta](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) - The Llama 3.1 8B Model serves as the LLM behind RAGify's intelligent responses.
- [Groq](https://wow.groq.com/why-groq/) - The LLM is hosted on Groq Language Processing Unit and inferences are provided through an API.
- [Streamlit](https://streamlit.io/) - Streamlit provides the technology to build and host the RAGify chatbot.
- **The Broader AI/ML Community** - A special thanks to the AI/ML community whose ongoing research and open-source contributions have laid the foundation for this project.
