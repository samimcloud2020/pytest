import requests
import pytest
from sentence_transformers import SentenceTransformer, util

# Inputs
QUESTION = "what is morden framework api"
GROUND_TRUTH_CONTEXT = "FastAPI is a modern framework for building APIs in Python."
GROUND_TRUTH_ANSWER = "FastAPI is a modern web framework for building APIs using Python."

API_URL = "http://54.82.142.56:8000/ask"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_sim(a, b):
    emb1 = embedding_model.encode(a, convert_to_tensor=True)
    emb2 = embedding_model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb1, emb2).item()

@pytest.mark.parametrize("question", [QUESTION])
def test_rag_metrics(question):
    response = requests.post(API_URL, json={"question": question})
    assert response.status_code == 200

    data = response.json()
    answer = data["answer"]
    retrieved_contexts = data["retrieved_contexts"]

    print("\n--- RAG METRICS ---")

    # Context Precision
    relevant_threshold = 0.6
    relevant_count = 0
    gt_emb = embedding_model.encode(GROUND_TRUTH_CONTEXT, convert_to_tensor=True)

    for ctx in retrieved_contexts:
        ctx_emb = embedding_model.encode(ctx, convert_to_tensor=True)
        sim = util.cos_sim(gt_emb, ctx_emb).item()
        print(f"[Precision] {ctx[:60]}... -> sim={sim:.3f}")
        if sim >= relevant_threshold:
            relevant_count += 1

    context_precision = relevant_count / len(retrieved_contexts)
    print(f"Context Precision: {context_precision:.2f}")
    assert context_precision >= 0.5

    # Context Recall (approximated: at least one context matches GT)
    recall_hit = any(util.cos_sim(gt_emb, embedding_model.encode(ctx, convert_to_tensor=True)).item() >= relevant_threshold for ctx in retrieved_contexts)
    context_recall = 1.0 if recall_hit else 0.0
    print(f"Context Recall: {context_recall:.2f}")
    assert context_recall == 1.0

    # Faithfulness: Answer should be semantically close to retrieved contexts
    avg_context_similarity = sum(cosine_sim(answer, ctx) for ctx in retrieved_contexts) / len(retrieved_contexts)
    print(f"Faithfulness (Answer vs Contexts Avg): {avg_context_similarity:.2f}")
    assert avg_context_similarity >= 0.5

    # Factual Correctness: Answer vs Ground Truth
    factual_correctness = cosine_sim(answer, GROUND_TRUTH_ANSWER)
    print(f"Factual Correctness: {factual_correctness:.2f}")
    assert factual_correctness >= 0.5

    # Answer Relevancy: Answer vs Question
    answer_relevancy = cosine_sim(answer, question)
    print(f"Answer Relevancy: {answer_relevancy:.2f}")
    assert answer_relevancy >= 0.3
