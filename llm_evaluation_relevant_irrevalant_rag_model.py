!pip install transformers sentence-transformers torch -qqq

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Load LLM Classifier (flan-t5 for relevance classification)
llm = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=10)

llm

# Load embedding model for cosine similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder

# Input
query = "What is the capital of France?"
retrieved_contexts = [
    "Paris is the capital and most populous city of France.",
    "Mount Everest is the highest mountain above sea level.",
    "The Eiffel Tower is located in Paris.",
    "Japan is located in East Asia."
]

### 1. LLM Classifier Approach
def classify_with_llm(query, context):
    prompt = f"Is the following context relevant to the question?\n\nQuestion: {query}\nContext: {context}\nAnswer yes or no:"
    print(prompt)
    response = llm(prompt)[0]['generated_text'].strip().lower()
    return "relevant" if "yes" in response else "nonrelevant"

results_llm = [(ctx, classify_with_llm(query, ctx)) for ctx in retrieved_contexts]
results_llm

### 2. Cosine Similarity Approach
def classify_by_cosine_similarity(query, contexts, threshold=0.5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    results = []
    for ctx in contexts:
        ctx_embedding = embedder.encode(ctx, convert_to_tensor=True)
        score = util.cos_sim(query_embedding, ctx_embedding).item()
        label = "relevant" if score >= threshold else "nonrelevant"
        results.append((ctx, label, round(score, 3)))
    return results

results_cosine = classify_by_cosine_similarity(query, retrieved_contexts)

results_cosine

### Print Results
print("\n📌 LLM Classification Results:")
for ctx, label in results_llm:
    print(f"{label.upper()}: {ctx}")

print("\n📌 Cosine Similarity Classification Results:")
for ctx, label, score in results_cosine:
    print(f"{label.upper()} ({score}): {ctx}")



