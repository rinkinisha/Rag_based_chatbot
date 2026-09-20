from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .embeddignd import create_embedding_model
from .vector_store import load_vector_store
from .retriever import retrieve_documents
from .llm import create_llm, generate_answer


app = FastAPI(
    title="Knowledge Base RAG API",
    description="RAG chatbot backend",
    version="1.0.0"
)


# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Initialize RAG components
# -----------------------------

embedding_model = create_embedding_model()

vector_store = load_vector_store(
    embedding_model
)

llm = create_llm()


# -----------------------------
# Request model
# -----------------------------

class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


# -----------------------------
# Health check
# -----------------------------

@app.get("/")
def home():
    return {
        "message": "RAG backend is running"
    }


# -----------------------------
# Chat endpoint
# -----------------------------

@app.post("/chat")
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "sources": []
        }


    # Retrieve relevant documents
    retrieved_documents = retrieve_documents(
        vector_store,
        question,
        k=3
    )


    # Generate grounded answer
    answer = generate_answer(
        llm,
        question,
        retrieved_documents ,
        request.history
    )


    # Prepare sources
    sources = []

    for document in retrieved_documents:

        metadata = document.metadata

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get("page")

        if page is not None:
            page = page + 1

        sources.append({
            "source": source,
            "page": page
        })


    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }