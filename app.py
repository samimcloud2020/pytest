from fastapi import FastAPI, Request
from pydantic import BaseModel
from model import generate_answer

app = FastAPI()
chat_history = []

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = generate_answer(request.question, chat_history)
    chat_history.append(f"User: {request.question}")
    chat_history.append(f"Bot: {answer}")
    return {"answer": answer, "chat_history": chat_history[-10:]}  # return recent history
