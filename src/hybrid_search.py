from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
def demo_ensemble_hybrid_search():
    """Hybrid search combining keyword (BM25) and semantic search."""

    print("=" * 60)
    print("ENSEMBLE/HYBRID RETRIEVER")
    print("Combines keyword (BM25) + semantic search")
    print("=" * 60)

    #create dummy document
    doc = [
    Document(
        page_content="Product SKU-7742X is our flagship router. "
                     "It supports Wi-Fi 6, dual-band connectivity, "
                     "and speeds up to 3 Gbps.",
        metadata={"type": "product", "category": "router"}
    ),
    Document(
        page_content="Product SKU-8821B is a budget wireless router. "
                     "It supports Wi-Fi 5, single-band connectivity, "
                     "and speeds up to 1 Gbps.",
        metadata={"type": "product", "category": "router"}
    ),
    Document(
        page_content="Product SKU-9912Z is a high-performance mesh router. "
                     "It supports Wi-Fi 6E, tri-band connectivity, "
                     "and covers large homes with multiple access points.",
        metadata={"type": "product", "category": "router"}
    ),
    Document(
        page_content="Our routers support WPA3 encryption for improved "
                     "wireless network security. WPA3 provides stronger "
                     "authentication compared to WPA2.",
        metadata={"type": "documentation", "category": "security"}
    ),
    Document(
        page_content="If customers experience slow internet speeds, "
                     "check network congestion, router placement, "
                     "firmware updates, and signal interference.",
        metadata={"type": "support", "category": "troubleshooting"}
    ),
    Document(
        page_content="Router firmware version 4.2.1 fixes frequent "
                     "connection drops and improves network stability.",
        metadata={"type": "documentation", "category": "firmware"}
    ),
    Document(
        page_content="The mesh networking feature allows multiple "
                     "access points to communicate with each other "
                     "and provide seamless Wi-Fi coverage.",
        metadata={"type": "documentation", "category": "networking"}
    ),
    Document(
        page_content="Product SKU-7742X comes with a 2-year warranty. "
                     "Customers can request replacement for manufacturing "
                     "defects during the warranty period.",
        metadata={"type": "product", "category": "warranty"}
    ),
    Document(
        page_content="Router installation requires connecting the WAN "
                     "port to the internet modem and configuring the "
                     "wireless network name and password.",
        metadata={"type": "documentation", "category": "installation"}
    ),
    Document(
        page_content="QoS (Quality of Service) allows administrators "
                     "to prioritize bandwidth for applications such as "
                     "video conferencing, gaming, and streaming.",
        metadata={"type": "documentation", "category": "networking"}
    )
]

    vector_store = Chroma.from_documents(
        documents=doc,
        embedding=embeddings,
        collection_name='hybrid_test'
    )

    # BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(doc)
    bm25_retriever.k = 3

    # Semantic retriever
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Ensemble combines both
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.7, 0.3],  # 40% keyword, 60% semantic
    )

    # Test queries
    # queries = [
    #     "PostgreSQL pgvector",  # Keyword-heavy (BM25 helps)
    #     "What database stores embeddings?",  # Semantic (vectors help)
    # ]
    queries = [
        # "ACID transactions",  # Keyword-heavy (BM25 helps)
        # "How do I store AI model outputs for later retrieval?",  # Semantic (vectors help)
        # "fast similarity lookup for embeddings",  # Mixed
        "SKU-7742X description"
         "Which router is suitable for a large house?",
         "WPA3 encryption",
         "firmware version 4.2.1",
        "How can I improve my wireless network security?",
        " What are the specifications of SKU-7742X?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        # Compare results
        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)
       # print(f"Ensamble top result: {ensemble_results}")
        print(f"BM25 top result: {bm25_results[0].page_content[:60]}...")
        print(f"Semantic top result: {semantic_results[0].page_content[:60]}...")
        print(f"Ensemble top result: {ensemble_results[0].page_content[:60]}...")


if __name__ == "__main__":
    demo_ensemble_hybrid_search()