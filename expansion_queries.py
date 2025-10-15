import os
from dotenv import load_dotenv
from pypdf import PdfReader
from helper_utils import word_wrap
from groq import Groq
import numpy as np
import umap
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from helper_utils import project_embeddings
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_key)

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter out empty strings
pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

query = "What was the total revenue of the year?"

results = chroma_collection.query(
    query_texts=[query], n_results=5
)
retrieved_documents = results["documents"][0]
# for document in retrieved_documents:
#     print(word_wrap(document))
#     print("\n")

# Function to generate multiple related queries using Groq API
def generate_multi_query(query, model="llama-3.3-70b-versatile"):
    prompt = """
    You are a knowledgeable research assistant. Your users are enquiring about an annual report.
    For the given question, propose upto five related questions to assist them in finding the information they need.
    Provide concise, single-topic questions (without compunding sentences) that cover various aspects of the topic.
    Ensure each question is complete and directly related to the original query.
    List each question on a separate line without numbering.
    """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": query,
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

original_query = (
    "What details can you provide about the factors that led to revenue growth?"
)

augmented_queries = generate_multi_query(original_query)

# First show the augmented queries
for query in augmented_queries:
    print("\n", query)

# Concatenate the original query with the augmented queries
joint_query = [original_query] + augmented_queries

# print("-------> \n\n", joint_query)

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Remove duplicates from the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# Output the unique retrieved documents
for i, document in enumerate(retrieved_documents):
    print(f"Query: {joint_query[i]}")
    print("")
    print("Results: ")
    for doc in documents:
        print(word_wrap(doc))
        print("")
    print("-" * 100)

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Visualize results in the embedding space
original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_query)

project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(augmented_query_embeddings, umap_transform)

retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

# Plot the projected query and retrieved documents in the embedding space
plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()
