from dotenv import load_dotenv
load_dotenv()

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
# torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 119.06 MiB is free. Including non-PyTorch memory, this process has 23.49 GiB memory in use. Of the allocated memory 23.24 GiB is allocated by PyTorch, and 1.17 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)