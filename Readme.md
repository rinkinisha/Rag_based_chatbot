# 🤖 Knowledge Base RAG Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions using information retrieved from a private knowledge base.

This project demonstrates a complete RAG pipeline using **LangChain, ChromaDB, Hugging Face embeddings, and Groq LLMs**.

The knowledge base currently contains documentation related to:

* React
* Node.js
* MongoDB

Instead of allowing the LLM to answer from its general knowledge, the system first retrieves relevant information from the knowledge base and then generates an answer using that retrieved context.

---

## 📌 Overview

Traditional LLM applications can generate answers using the knowledge learned during model training. However, they may not know about private documents or specific knowledge bases.

This project solves that problem using **Retrieval-Augmented Generation**.

The basic flow is:

```text
User Question
      ↓
Query Embedding
      ↓
Semantic Search
      ↓
ChromaDB
      ↓
Relevant Documents
      ↓
Context + Question
      ↓
Groq LLM
      ↓
Grounded Answer
      ↓
Sources
```

If the requested information is not available in the knowledge base, the chatbot is instructed to respond:

```text
I don't know based on the provided knowledge base.
```

This prevents the chatbot from intentionally generating information that is not supported by the provided documents.

---

# ✨ Features

* 📄 Load knowledge from PDF documents
* ✂️ Split documents into smaller chunks
* 🔢 Generate vector embeddings
* 🗄️ Store embeddings in ChromaDB
* 🔎 Perform semantic similarity search
* 🧠 Generate answers using a Groq-hosted LLM
* 📚 Ground answers using retrieved knowledge-base context
* 💬 Maintain conversation history
* 📑 Return source document and page information
* 🚫 Handle questions outside the available knowledge base
* ⚡ FastAPI REST API
* 📖 Automatic Swagger API documentation
* ⚛️ React frontend for interacting with the chatbot

---

# 🛠️ Tech Stack

| Technology            | Purpose                                |
| --------------------- | -------------------------------------- |
| Python                | Backend and RAG pipeline               |
| LangChain             | Document processing and RAG components |
| LangChain Community   | PDF document loading                   |
| LangChain Chroma      | ChromaDB integration                   |
| Hugging Face          | Text embedding model                   |
| Sentence Transformers | Embedding generation                   |
| ChromaDB              | Vector database                        |
| Groq                  | LLM inference                          |
| FastAPI               | Backend API                            |
| Uvicorn               | ASGI server                            |
| React                 | Frontend                               |
| Vite                  | Frontend development                   |
| JavaScript            | Frontend logic                         |
| dotenv                | Environment variable management        |

---

# 🏗️ Project Architecture

```text
                    ┌──────────────────┐
                    │   PDF Documents  │
                    │ React / Node /   │
                    │     MongoDB      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  PDF Loader      │
                    │   PyPDFLoader    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Chunking      │
                    │                  │
                    │ 500 characters   │
                    │ 50 overlap       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Embeddings     │
                    │ Hugging Face     │
                    │ all-MiniLM-L6-v2 │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │  Vector Store    │
                    └────────┬─────────┘
                             │
                    User Question
                             │
                             ▼
                    ┌──────────────────┐
                    │ Semantic Search  │
                    │     Top-K = 3    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Retrieved        │
                    │ Documents        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Groq LLM      │
                    │  GPT-OSS-20B     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Grounded Answer  │
                    │   + Sources      │
                    └──────────────────┘
```

---

# 📂 Project Structure

```text
KB-rag-chatbot/
│
├── data/
│   ├── React.pdf
│   ├── Node.pdf
│   └── MongoDB.pdf
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddignd.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── llm.py
│   ├── ingest.py
│   └── main.py
│
├── chroma_db/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Chat.jsx
│   │   │   └── Message.jsx
│   │   │
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   └── package.json
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

> **Note:** `embeddignd.py` is intentionally shown with the current filename used in the project.

---

# 🔄 How RAG Works in This Project

## 1. Document Ingestion

PDF documents are loaded using LangChain's `PyPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
```

Each PDF page becomes a LangChain `Document` containing:

```text
page_content
metadata
```

The metadata contains information such as the source file and page number.

---

## 2. Document Chunking

Large documents are divided into smaller pieces before generating embeddings.

This project uses:

```text
Chunk Size   → 500
Chunk Overlap → 50
```

The project uses LangChain's:

```python
RecursiveCharacterTextSplitter
```

Example:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(documents)
```

### Why chunking?

Instead of storing an entire document as one large vector, smaller chunks allow the retrieval system to find more relevant sections.

---

# 🧮 Embeddings

Text cannot be directly compared semantically by a vector database.

Therefore, each chunk is converted into a numerical vector called an **embedding**.

This project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain's Hugging Face integration.

```python
from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings
```

Conceptually:

```text
"React is a JavaScript library"
                ↓
        Embedding Model
                ↓
[0.021, -0.183, 0.472, ...]
```

The resulting vectors represent the semantic meaning of the text.

---

# 🗄️ ChromaDB

The generated embeddings are stored in **ChromaDB**.

The project uses:

```text
Collection:
knowledge_base

Storage:
./chroma_db
```

ChromaDB allows the application to search for documents that are semantically similar to a user's question.

The vector store is created using LangChain's Chroma integration.

```python
from langchain_chroma import Chroma
```

---

# 🔎 Semantic Retrieval

When a user asks a question, the question is converted into an embedding.

For example:

```text
"What is React state?"
```

The system searches ChromaDB for the most semantically similar chunks.

Currently:

```text
Top-K = 3
```

The three most relevant documents are retrieved and passed to the LLM as context.

```python
results = vector_store.similarity_search(
    query,
    k=3
)
```

The important difference between keyword search and semantic search is that semantic search focuses on **meaning**, not just exact word matching.

---

# 🧠 LLM and Grounding

After retrieving relevant documents, the system sends the following to the LLM:

```text
Conversation History
+
Knowledge Base Context
+
Current Question
```

The LLM is instructed to answer using only the supplied knowledge-base context.

The current LLM configuration uses:

```text
Provider: Groq
Model: openai/gpt-oss-20b
Temperature: 0
```

The grounding instruction is conceptually:

```text
Answer the user's question using ONLY
the provided knowledge-base context.

Do not make up information.

If the answer cannot be found in the
knowledge-base context, say:

"I don't know based on the provided
knowledge base."
```

This is the core grounding mechanism of the application.

---

# 💬 Conversation History

The chatbot also supports conversation history.

For example:

```text
User:
What is React?

Assistant:
React is a JavaScript library...

User:
What are its components?
```

The previous conversation is passed to the backend so the LLM can understand references such as:

```text
it
its
they
this
that
```

The frontend sends history as part of the request:

```json
{
  "question": "What are its components?",
  "history": [
    {
      "role": "user",
      "content": "What is React?"
    },
    {
      "role": "assistant",
      "content": "React is a JavaScript library..."
    }
  ]
}
```

---

# 📚 Source Information

The API also returns information about the documents used during retrieval.

Example response:

```json
{
  "question": "What is React state?",
  "answer": "React state is ...",
  "sources": [
    {
      "source": "data/React.pdf",
      "page": 4
    }
  ]
}
```

This makes the answer more transparent by showing where the retrieved information came from.

---

# 🚫 Handling Questions Outside the Knowledge Base

The chatbot is designed to avoid answering questions using unrelated general knowledge.

For example, if the knowledge base contains information about:

```text
React
Node.js
MongoDB
```

and the user asks:

```text
Who was the first person to walk on the Moon?
```

the system should not invent an answer from the LLM's general knowledge.

Instead, the grounding instruction tells the model to respond:

```text
I don't know based on the provided knowledge base.
```

This is an important property of the RAG application because the goal is to answer from the provided knowledge base rather than behave like a general-purpose chatbot.

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project:

```bash
cd KB-rag-chatbot
```

---

# 🐍 Backend Setup

## 2. Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate it:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` to GitHub.

Your `.gitignore` should contain:

```gitignore
venv/
.env
__pycache__/
chroma_db/
```

---

# 📄 Add Knowledge Base Documents

Place your PDF documents inside:

```text
data/
```

For example:

```text
data/
├── React.pdf
├── Node.pdf
└── MongoDB.pdf
```

You can replace or add PDFs when you want to change the knowledge base.

---

# ⚙️ Build the Vector Database

Whenever the knowledge-base documents change, run the ingestion pipeline:

```bash
python3 src/ingest.py
```

The ingestion process:

```text
PDFs
 ↓
Load Documents
 ↓
Split into Chunks
 ↓
Generate Embeddings
 ↓
Store in ChromaDB
```

You should see output similar to:

```text
Found 3 PDF files

Loading: data/React.pdf
Pages loaded: ...

Loading: data/Node.pdf
Pages loaded: ...

Loading: data/MongoDB.pdf
Pages loaded: ...

Total documents/pages: ...
Total chunks: ...

Embedding model created

Ingestion completed successfully!
ChromaDB created at: chroma_db
```

---

# 🚀 Start the Backend

Run:

```bash
uvicorn src.main:app --reload
```

The backend will run at:

```text
http://127.0.0.1:8000
```

---

# 📖 API Documentation

FastAPI automatically generates interactive API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🔌 API Endpoints

## GET `/`

Checks whether the backend is running.

Example response:

```json
{
  "message": "RAG backend is running"
}
```

---

## POST `/chat`

Sends a question to the RAG chatbot.

### Request

```json
{
  "question": "What is React state?",
  "history": []
}
```

### Response

```json
{
  "question": "What is React state?",
  "answer": "React state is ...",
  "sources": [
    {
      "source": "data/React.pdf",
      "page": 4
    }
  ]
}
```

---

# ⚛️ Frontend Setup

The project also contains a React frontend.

Move into the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The Vite development server will provide the frontend URL in the terminal.

The frontend communicates with:

```text
http://127.0.0.1:8000/chat
```

---

# 🧪 Testing the RAG System

The project should be tested with two categories of questions.

## 1. Questions Covered by the Knowledge Base

Examples:

```text
What is React?

What is React state?

What is Node.js?

What is MongoDB?
```

Expected behavior:

```text
Retrieve relevant chunks
        ↓
Generate grounded answer
        ↓
Return source information
```

---

## 2. Questions Not Covered by the Knowledge Base

Examples:

```text
What is the capital of France?

Who discovered gravity?

What is the population of Japan?
```

Expected behavior:

```text
Information not found
        ↓
Do not invent an answer
        ↓
"I don't know based on the provided knowledge base."
```

This distinction is important when evaluating whether the RAG system is actually grounded in its source documents.

---

# 🧩 Core Modules

## `ingestion.py`

Responsible for loading PDF documents.

```text
PDF → LangChain Documents
```

---

## `chunking.py`

Responsible for splitting documents into smaller chunks.

```text
Documents → Chunks
```

---

## `embeddignd.py`

Responsible for creating the embedding model.

```text
Text → Vector
```

---

## `vector_store.py`

Responsible for creating and loading the ChromaDB vector store.

```text
Chunks + Embeddings → ChromaDB
```

---

## `retriever.py`

Responsible for semantic retrieval.

```text
Question → Relevant Documents
```

---

## `llm.py`

Responsible for:

* Creating the Groq client
* Building the grounded prompt
* Passing context to the LLM
* Generating the final answer

---

## `ingest.py`

Runs the complete document ingestion pipeline.

```text
PDF
 ↓
Loader
 ↓
Chunking
 ↓
Embeddings
 ↓
ChromaDB
```

This script should be run when the knowledge base needs to be rebuilt.

---

## `main.py`

Provides the FastAPI application.

```text
Frontend
   ↓
POST /chat
   ↓
Retriever
   ↓
ChromaDB
   ↓
Retrieved Context
   ↓
LLM
   ↓
Answer + Sources
```

---

# 🧠 Important RAG Concepts Demonstrated

This project demonstrates several important concepts:

### Retrieval-Augmented Generation

The LLM receives retrieved information before generating an answer.

```text
Question
+
Retrieved Context
        ↓
       LLM
        ↓
     Answer
```

### Embeddings

Text is represented as numerical vectors so semantic similarity can be calculated.

### Vector Database

ChromaDB stores and searches vector representations of document chunks.

### Semantic Search

Documents are retrieved based on semantic similarity rather than only exact keyword matches.

### Top-K Retrieval

The system currently retrieves:

```text
K = 3
```

relevant chunks for each query.

### Grounding

The LLM is instructed to use the retrieved knowledge-base context instead of inventing information.

### Source Attribution

The API returns the source document and page associated with retrieved content.

### Conversation History

Previous messages are provided to the LLM so follow-up questions can be understood in context.

---

# 🔐 Security Considerations

The following files should never be committed:

```text
.env
venv/
chroma_db/
__pycache__/
```

Especially:

```text
GROQ_API_KEY
```

must remain private.

A production application should also replace the development CORS configuration with an explicit list of trusted frontend origins.

---

# 📈 Current RAG Pipeline

The complete implementation can be summarized as:

```text
                 OFFLINE / INGESTION
                 ===================

             React.pdf
             Node.pdf
             MongoDB.pdf
                  │
                  ▼
            PyPDFLoader
                  │
                  ▼
          Document Chunking
          500 / 50 overlap
                  │
                  ▼
       Hugging Face Embeddings
                  │
                  ▼
              ChromaDB
                  │
                  │
                  │
                  ▼
              KNOWLEDGE BASE


                 ONLINE / QUERY
                 ==============

              User Question
                    │
                    ▼
             Semantic Search
                    │
                    ▼
              ChromaDB
                    │
                    ▼
              Top 3 Chunks
                    │
                    ▼
          Context + Question
          + Conversation History
                    │
                    ▼
              Groq LLM
                    │
                    ▼
           Grounded Response
                    │
             ┌──────┴──────┐
             ▼             ▼
           Answer       Sources
```

---

# 🎯 Project Goals

The project demonstrates how to build a practical RAG system from the ground up using:

* Document ingestion
* Document chunking
* Embeddings
* Vector databases
* Semantic retrieval
* LLM integration
* Grounding
* Source attribution
* Conversation history
* REST API integration

The goal is not simply to connect an LLM to an application, but to understand the complete flow of how **private knowledge can be retrieved and supplied to an LLM at query time**.

---

# 🔮 Future Improvements

Possible future improvements include:

* Retrieval similarity thresholds
* Better handling of ambiguous follow-up questions
* Conversation-aware query rewriting
* Streaming LLM responses
* Improved source citation UI
* Authentication and user accounts
* Persistent chat sessions
* More advanced evaluation metrics
* Experimentation with different chunk sizes and overlap values
* Production deployment
* More granular document metadata filtering

---

# 📚 Learning Outcomes

After completing this project, the following concepts are demonstrated:

```text
                    RAG
                     │
        ┌────────────┼────────────┐
        │            │            │
   Ingestion     Retrieval       Generation
        │            │            │
        ▼            ▼            ▼
    PDF Loader   Embeddings      LLM
    Chunking     ChromaDB       Prompt
        │        Semantic       Grounding
        │        Search
        └────────────┼────────────┘
                     │
                     ▼
              RAG Application
```

---

# 👩‍💻 Author

**Rinki Nisha**

Full-Stack Developer | React.js | Node.js | AI/LLM Integration

* GitHub: https://github.com/rinkinisha
* LinkedIn: https://www.linkedin.com/in/rinki-nisha-8b762b332/

---

# ⭐ Project Summary

**Knowledge Base RAG Chatbot** is a full-stack Retrieval-Augmented Generation application that allows users to ask questions about a predefined collection of documents.

The system processes PDF documents, divides them into chunks, converts those chunks into embeddings, stores them in ChromaDB, retrieves relevant information using semantic search, and provides that information to a Groq LLM to generate grounded responses.

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
ChromaDB
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
Grounded Answer + Sources
```

**Built to understand RAG — not just use it.**
