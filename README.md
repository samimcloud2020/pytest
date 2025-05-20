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
           
