from huggingface_hub import login
import os

result = login(token=os.environ['HUGGING_FACE_HUB_TOKEN'])
print(result)



from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct", device=0)
outs = pipe(messages)
print(outs)