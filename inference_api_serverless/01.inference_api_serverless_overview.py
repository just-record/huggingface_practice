from dotenv import load_dotenv
load_dotenv()

import requests
import os

API_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# data = query("Can you please let us know more details about your ")
data = query({
    "inputs": "Can you please let us know more details about your "
})

print(f'data: {data}')