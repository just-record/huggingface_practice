from transformers import AutoTokenizer

# tokenizer - 사전 학습에 사용한 vocab을 다운로드
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

long_sentence = "Don't think he knows about second breakfast, Pip." * 100

batch_sentences = [
    "But what about second breakfast?",
    long_sentence,
    "What about elevensies?",
]

### truncation 처리 안함
encoded_input = tokenizer(batch_sentences, padding=True)
print(len(encoded_input['input_ids'][0]))
print(len(encoded_input['input_ids'][1]))
print(len(encoded_input['input_ids'][2]))

print('='*30)

### truncation 처리
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
print(len(encoded_input['input_ids'][0]))
print(len(encoded_input['input_ids'][1]))
print(len(encoded_input['input_ids'][2]))