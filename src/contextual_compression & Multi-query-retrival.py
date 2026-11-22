from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.stores import InMemoryStore
from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.documents import Document

# Sample knowledge base for demos
TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]

def create_base_vectorstore():
    """Create a basic vector store for demos."""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
def demo_contextual_compression():
    """Contextual Compression extracts only relevant parts."""

    print("=" * 60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant content from documents")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Create compressor
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    query = "What frameworks exist for building LLM applications?"

    print(f"\nQuery: {query}")

    # Without compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    # With compression
    # With compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content}\n")

def demo_multi_query_retriever():
    """Multi-Query Retriever generates multiple query perspectives."""

    print("=" * 60)
    print("MULTI-QUERY RETRIEVER")
    print("Generates multiple perspectives on your question")
    print("=" * 60)

    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Create multi-query retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}), llm=llm
    )

    query = "Why Pinecone is used?"

    print(f"\nOriginal Query: {query}")
    print("\nThe retriever will generate multiple query variations...")
    print("(Check INFO logs above for generated queries)\n")

    # Retrieve documents
    docs = retriever.invoke(query)

    print(f"Retrieved {len(docs)} unique documents:")
    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic', 'N/A')}] {doc.page_content[:100]}..."
        )


if __name__ =="__main__":
    #demo_contextual_compression()
    demo_multi_query_retriever()