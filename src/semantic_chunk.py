from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
Documents = '''
# Artificial Intelligence

Artificial Intelligence is transforming industries by enabling machines to perform tasks that traditionally required human intelligence. Machine learning is a subset of artificial intelligence that allows systems to learn patterns from historical data. Deep learning uses neural networks with multiple layers to process complex information such as images, audio, and natural language.

Large Language Models are trained on massive datasets and can generate human-like responses. Retrieval Augmented Generation combines information retrieval with language generation. Instead of relying entirely on the model's internal knowledge, RAG retrieves relevant external documents and provides them as context to the language model.

# Space Exploration

The solar system consists of eight planets orbiting the Sun. Jupiter is the largest planet, while Mercury is the closest planet to the Sun. Saturn is famous for its prominent ring system, which consists primarily of ice particles, rocky debris, and dust.

NASA's Voyager missions have provided valuable information about the outer planets. The James Webb Space Telescope observes distant galaxies, stars, and planetary systems using infrared technology. Scientists use these observations to understand how galaxies formed and evolved over billions of years.

# Database Management

Relational databases organize information into tables containing rows and columns. SQL is commonly used to create, retrieve, update, and delete records. PostgreSQL supports advanced features such as indexing, transactions, JSON operations, and complex queries.

NoSQL databases provide alternative data models for applications with flexible schemas and distributed workloads. MongoDB stores information in document-oriented structures, while Redis is frequently used as an in-memory key-value store.

Database indexing improves query performance by reducing the amount of data that must be scanned. However, indexes require additional storage and can increase the overhead of write operations.

# Climate Change

Climate change refers to long-term changes in global temperatures and weather patterns. Human activities, particularly the burning of fossil fuels, increase greenhouse gas concentrations in the atmosphere.

Carbon dioxide and methane are two important greenhouse gases. Renewable energy sources such as solar, wind, and hydropower can help reduce dependence on fossil fuels. Governments and organizations are investing in sustainable transportation, energy-efficient buildings, and improved waste management.

'''

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, chunk_overlap = 50,
    separators=["/n/n", "/n", " ", ""]
)
chunks = splitter.split_text(Documents)

for i, chunk in enumerate(chunks):
    print(f"chunks...{i+1} and length is : {len(chunk)}")
    print(chunk[:100]+"..." if len(chunk)>100 else  chunk)


semantic_chunk = SemanticChunker(embeddings,
                                 breakpoint_threshold_amount="90%",
                                 breakpoint_threshold_type="percentile")

sem_chunk = semantic_chunk.split_text(Documents)
for i, sem_ch in enumerate(sem_chunk):
     print(f"chunks...{i+1} and length is : {len(sem_ch)}")
     print(chunk[:100]+"..." if len(sem_ch)>100 else  sem_ch)
    