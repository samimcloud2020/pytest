from fastapi import FastAPI
from pydantic import BaseModel
from model import generate_answer

app = FastAPI()
chat_history = []

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer, retrieved_contexts = generate_answer(request.question)

    chat_history.append(f"User: {request.question}")
    chat_history.append(f"Bot: {answer}")

    return {
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "chat_history": chat_history[-10:]
    }

@app.delete("/clear_history")
async def clear_history():
    chat_history.clear()
    return {"message": "Chat history cleared."}
