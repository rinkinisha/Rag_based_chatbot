from pathlib import Path
import shutil

from ingestion import load_pdf
from chunking import split_documents
from embeddignd import create_embedding_model
from vector_store import create_vector_store


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")


def main():

    # Remove old vector database
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("Old ChromaDB removed")

    all_documents = []

    # Find all PDFs inside data/
    pdf_files = list(DATA_DIR.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files")

    # Load every PDF
    for pdf_file in pdf_files:

        print(f"\nLoading: {pdf_file}")

        documents = load_pdf(str(pdf_file))

        all_documents.extend(documents)

        print(f"Pages loaded: {len(documents)}")

    print("\nTotal documents/pages:", len(all_documents))

    # Split all documents into chunks
    chunks = split_documents(all_documents)

    print("Total chunks:", len(chunks))

    # Create embedding model
    embedding_model = create_embedding_model()

    print("Embedding model created")

    # Store chunks in ChromaDB
    create_vector_store(
        chunks,
        embedding_model
    )

    print("\nIngestion completed successfully!")
    print("ChromaDB created at:", CHROMA_DIR)


if __name__ == "__main__":
    main()