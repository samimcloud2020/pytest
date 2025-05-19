import os
import torch
import faiss
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# Load documents
def load_documents(file_path="data.txt"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

documents = load_documents()

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_embeddings = embedding_model.encode(documents, convert_to_tensor=False)

# FAISS index
dimension = document_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Hugging Face Token (used securely)
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    raise EnvironmentError("HF_API_TOKEN is not set in environment.")

# Load Mistral from Hugging Face using token
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

def generate_answer(question: str, chat_history: list[str]):
    # Retrieve relevant context
    question_embedding = embedding_model.encode([question])[0]
    _, I = index.search(question_embedding.reshape(1, -1), k=2)
    retrieved_contexts = [documents[i] for i in I[0]]

    # Construct prompt
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
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()
