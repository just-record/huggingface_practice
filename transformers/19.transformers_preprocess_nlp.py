from transformers import AutoTokenizer

# tokenizer - 사전 학습에 사용한 vocab을 다운로드
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

# 토큰으로 분리하고 숫자로 변환
encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)
# {'input_ids': [101, 2091, 1136, 1143, 13002, 1107, 1103, 5707, 1104, 16678, 1116, 117, 1111, 1152, 1132, 11515, 1105, 3613, 1106, 4470, 119, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

print('='*30)

# 숫자로 변환된 토큰을 다시 문자열로 변환
decode = tokenizer.decode(encoded_input["input_ids"])
print(decode)
# [CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]


print('='*30)

### 여러 문장을 처리하는 경우
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)
# {
#     'input_ids': [
#         [101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
#         [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
#         [101, 1327, 1164, 5450, 23434, 136, 102]
#     ], 
#  'token_type_ids': [
#         [0, 0, 0, 0, 0, 0, 0, 0], 
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#         [0, 0, 0, 0, 0, 0, 0]
#     ], 
#  'attention_mask': [
#         [1, 1, 1, 1, 1, 1, 1, 1], 
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
#         [1, 1, 1, 1, 1, 1, 1]
#     ]
# }
