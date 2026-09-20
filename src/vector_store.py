from langchain_chroma import Chroma


COLLECTION_NAME = "knowledge_base"
PERSIST_DIRECTORY = "./chroma_db"


def create_vector_store(chunks, embedding_model):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    vector_store.add_documents(chunks)

    return vector_store


def load_vector_store(embedding_model):
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store