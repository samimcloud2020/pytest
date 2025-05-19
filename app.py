from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from model import generate_answer

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    chat_history: List[str] = []

@app.post("/ask")
def ask(request: AskRequest):
    answer = generate_answer(request.question, request.chat_history)
    return {"answer": answer}
