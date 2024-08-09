# from huggingface_hub import login
# login()
# cmd에 access token 입력 필요


from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
import os

result = login(token=os.environ['HUGGING_FACE_HUB_TOKEN'])
print(result)
# Token is valid (permission: write).
# Your token has been saved to /home/dev/.cache/huggingface/token
# Login successful
# None