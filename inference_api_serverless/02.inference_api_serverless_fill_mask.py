from dotenv import load_dotenv
load_dotenv()

import requests
import os

API_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')

headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/bert-base-uncased"

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

data = query({"inputs": "The answer to the universe is [MASK]."})
# - 파라미터
#     - inputs (필수): 채워질 문자열로, [MASK] 토큰을 포함해야 합니다 (정확한 마스크 이름은 모델 카드에서 확인하세요)

print(f'data: {data}')
# data: [
# {'score': 0.1696401685476303, 'token': 2053, 'token_str': 'no', 'sequence': 'the answer to the universe is no.'}, 
# {'score': 0.07344774901866913, 'token': 2498, 'token_str': 'nothing', 'sequence': 'the answer to the universe is nothing.'}, 
# {'score': 0.0580325648188591, 'token': 2748, 'token_str': 'yes', 'sequence': 'the answer to the universe is yes.'}, 
# {'score': 0.043957922607660294, 'token': 4242, 'token_str': 'unknown', 'sequence': 'the answer to the universe is unknown.'}, 
# {'score': 0.04015738517045975, 'token': 3722, 'token_str': 'simple', 'sequence': 'the answer to the universe is simple.'}]

# - 반환값
#   - sequence: 모델에 실행된 실제 토큰 시퀀스 (특수 토큰을 포함할 수 있음)
#   - score: 이 토큰에 대한 확률
#   - token: 토큰의 ID
#   - token_str: 토큰의 문자열 표현  