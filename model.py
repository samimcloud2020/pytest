import os
import faiss
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load data.txt
def load_documents(file_path="data.txt"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

documents = load_documents()

# Load sentence embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents, convert_to_tensor=False)

# FAISS index
dimension = document_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Falcon 1B model for low-resource EC2
model_name = "tiiuae/falcon-rw-1b"
offload_folder = "./offload"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=offload_folder,
    trust_remote_code=True
)

def generate_answer(question: str, chat_history: list[str]):
    # Retrieve top 5 documents
    question_embedding = embedding_model.encode([question])[0]
    _, I = index.search(question_embedding.reshape(1, -1), k=5)
    retrieved_contexts = [documents[i] for i in I[0]]

    context = "\n".join(retrieved_contexts)
    history = "\n".join(chat_history)
    prompt = f"""You are a helpful assistant. Use the context below to answer the user's question.

Context:
{context}

Chat History:
{history}

Question:
{question}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()
