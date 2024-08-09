from dotenv import load_dotenv
load_dotenv()

import requests
import os

API_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query(
    {
        "inputs": {
            "question": "What's my name?",
            "context": "My name is Clara and I live in Berkeley.",
        }
    }
)

print(f'data: {data}')
# data: {'score': 0.9326565265655518, 'start': 11, 'end': 16, 'answer': 'Clara'}