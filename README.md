apt update -y

apt install python3.12-venv

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

export HF_API_TOKEN=""

uvicorn app:app --host 0.0.0.0 --port 8000


pip install ragas datasets evaluate

pip install pytest requests

pip install ragas evaluate datasets

pytest test_context_precision.py -s

python -m pytest test_context_precision.py -s     <----------------its worked
           
context_precision:-----------------------

(venv) root@ip-172-31-32-19:~/pytest# python -m pytest test_context_precision.py -s

========================================================================= test session starts 
=========================================================================

platform linux -- Python 3.12.3, pytest-8.3.5, pluggy-1.6.0

rootdir: /root/pytest

plugins: langsmith-0.3.42, anyio-4.9.0

collected 1 item                                                                                                                                                      

test_context_precision.py Context: FastAPI is a modern, fast (high-performance), web framework ... -> Similarity: 0.937

Context: FastAPI is a modern, fast (high-performance), web framework ... -> Similarity: 0.937

Context: Automatic Documentation: FastAPI generates interactive API d... -> Similarity: 0.695

Context: Dependency Injection: FastAPI supports dependency injection,... -> Similarity: 0.606

Context: Middleware: We can easily add middleware to your FastAPI app... -> Similarity: 0.649

Context Precision: 1.00
.

========================================================================== 1 passed in 7.57s ==========================================================================

