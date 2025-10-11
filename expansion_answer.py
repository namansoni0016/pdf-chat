from pypdf import PdfReader
import os
from helper_utils import project_embeddings, word_wrap
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import umap
import matplotlib.pyplot as plt

# Loading environment variables from .env file
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_key)

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter the empty strings
pdf_texts = [text for text in pdf_texts if text]
# print(word_wrap(pdf_texts[0], width=100))



# Split the text into smaller chunks 
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
# print(word_wrap(character_split_texts[10]))
print(f"\nTotal Chunks: {len(character_split_texts)}")



# Using SentenceTransformersTokenTextSplitter to split text based on token count of 256
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)
# print(word_wrap(token_split_texts[10]))
print(f"\nTotal Chunks after Token Split: {len(token_split_texts)}")


embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[0]]))
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)

# Extract the embeddings of the token_split_texts and add them to the ChromaDB collection
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()

query = "What was the total revenue of the year?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_docs = results["documents"][0]

# for document in retrieved_docs:
#     print(word_wrap(document))
#     print("\n")

# Passing the query through LLM and generating an answer
def augment_query_generated(query, model="llama-3.3-70b-versatile"):
    prompt = """You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report."""
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        { "role": "user", "content": query },
    ]
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "What was the total profit for the year, and how does it compare to the previous year?"
hypothetical_answer = augment_query_generated(original_query)

# Adding the hypothetical answer to the query so that we can add it in vector DB
joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
# Now storing the joint_query in the vector DB
results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_docs = results["documents"][0]
# for document in retrieved_docs:
#     print(word_wrap(document))
#     print("")

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augment_query_embedding = embedding_function([joint_query])

project_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)

project_augmented_query_embedding = project_embeddings(
    augment_query_embedding, umap_transform
)

project_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

# Plotting the embeddings using matplotlib
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)

plt.scatter(
    project_retrieved_embeddings[:, 0],
    project_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)

plt.scatter(
    project_original_query_embedding[:, 0],
    project_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.scatter(
    project_augmented_query_embedding[:, 0],
    project_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()