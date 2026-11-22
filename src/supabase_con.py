import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

def connection_to_supabase():
    database_url = os.getenv("SUPABASE_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "Set SUPABASE_DATABASE_URL or DATABASE_URL in your environment or .env file."
        )

    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    vector_store = PGVector(
        embeddings=embeddings,
        connection=database_url,
        collection_name="prod_docs",
    )

    return vector_store

def verify_connection(vector_store):
    from langchain_core.documents import Document
    docs = Document(
        page_content="This is the sample document to test supabase vector db/pg vector",
        metadata={"source": "test"},
    )

    ids =  vector_store.add_documents(documents=[docs])
    print(f"Added test document {ids[0]}")

    result = vector_store.similarity_search(
        query="sample document to test Supabase vector database",
        k=1,
    )
    for document in result:
        print(f"Found similar document: {document.page_content}")

    vector_store.delete(ids=ids)
    print("cleanup-completed")


if __name__ =="__main__":
    vector_store = connection_to_supabase()
    verify_connection(vector_store=vector_store)