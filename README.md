apt update -y
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export HF_API_TOKEN=""
uvicorn app:app --host 0.0.0.0 --port 8000
