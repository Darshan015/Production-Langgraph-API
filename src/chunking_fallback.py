"""
Semantic chunking with a recursive fallback.

Reconstructed from the course screenshot (prod_ready.py, smart_chunker).
The lines hidden in the screenshot (function parameters) are my best guess.

Install:
    uv pip install langchain-experimental langchain-openai langchain-text-splitters python-dotenv
"""

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


def _recursive_fallback(text: str, chunk_size: int = 500) -> list[str]:
    """Recursive character splitting: paragraph -> line -> word -> character."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,  # 10% overlap
    )
    print("Executing recursive chunking strategy")
    return splitter.split_text(text)


def smart_chunker(
    text: str,
    use_semantic: bool = True,
    max_chunk_size: int = 150,
    fallback_chunk_size: int = 100,
) -> list[str]:
    """
    Try semantic chunking first. Fall back to recursive splitting when:
      1. any semantic chunk is larger than max_chunk_size, or
      2. semantic chunking raises an error (e.g. embedding API failure).
    """
    if use_semantic:
        try:
            chunker = SemanticChunker(
                embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=90,  # split at the top 10% biggest meaning shifts
            )
            chunks = chunker.split_text(text)

            # Validate chunks aren't too large
            if any(len(c) > max_chunk_size for c in chunks):
                # Fallback to recursive for oversized chunks
                return _recursive_fallback(text, fallback_chunk_size)

            return chunks

        except Exception as e:
            print(f"Semantic chunking failed: {e}, using fallback")
            return _recursive_fallback(text, fallback_chunk_size)

    return _recursive_fallback(text, fallback_chunk_size)


if __name__ == "__main__":
    sample = """Machine learning is a subset of artificial intelligence. It lets systems learn from data.

Supervised learning uses labeled data. The model learns to map inputs to outputs.

Cats are popular pets. They sleep for most of the day and enjoy chasing toys."""

    result = smart_chunker(sample)
    print(f"Produced {len(result)} chunks")
    for i, chunk in enumerate(result):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---\n{chunk}")