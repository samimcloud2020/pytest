import requests
import pytest
from sentence_transformers import SentenceTransformer, util

# Question and expected relevance
question = "what is morden framework api"
ground_truth = "FastAPI is a modern framework for building APIs in Python."

# API URL
API_URL = "http://54.82.142.56:8000/ask"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

@pytest.mark.parametrize("question", [question])
def test_manual_context_precision(question):
    response = requests.post(API_URL, json={"question": question})
    assert response.status_code == 200

    data = response.json()
    retrieved_contexts = data["retrieved_contexts"]

    # Compute precision: how many contexts are relevant?
    relevant_threshold = 0.6  # Cosine similarity
    relevant_count = 0

    q_emb = embedding_model.encode(ground_truth, convert_to_tensor=True)
    for ctx in retrieved_contexts:
        ctx_emb = embedding_model.encode(ctx, convert_to_tensor=True)
        sim = util.cos_sim(q_emb, ctx_emb).item()
        print(f"Context: {ctx[:60]}... -> Similarity: {sim:.3f}")
        if sim >= relevant_threshold:
            relevant_count += 1

    precision = relevant_count / len(retrieved_contexts)
    print(f"\nContext Precision: {precision:.2f}")

    assert precision >= 0.5, f"Low precision: {precision:.2f}"
