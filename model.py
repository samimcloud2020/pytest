import os
from pathlib import Path

import faiss
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# Load documents from file
def load_documents(file_path="data.txt"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

documents = load_documents()

# Load embedding model and encode documents
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents, convert_to_tensor=False)

# Build FAISS index
dimension = document_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Load CPU-friendly LLM and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to("cpu")

# Function to generate answer
def generate_answer(question: str):
    # Retrieve top 5 relevant documents
    question_embedding = embedding_model.encode([question])[0]
    _, indices = index.search(question_embedding.reshape(1, -1), k=5)
    retrieved_contexts = [documents[i] for i in indices[0]]

    # Create prompt with context + question
    context = "\n".join(retrieved_contexts)
    prompt = f"""Answer the question using the following context.

Context:
{context}

Question:
{question}"""

    # Handle pad token
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id

    # Tokenize and generate
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=200,
        pad_token_id=pad_token_id
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer.strip(), retrieved_contexts
