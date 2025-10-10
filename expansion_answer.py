from pypdf import PdfReader
import os
from helper_utils import word_wrap
from groq import Groq
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Loading environment variables from .env file
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

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
print(word_wrap(token_split_texts[10]))
print(f"\nTotal Chunks after Token Split: {len(token_split_texts)}")