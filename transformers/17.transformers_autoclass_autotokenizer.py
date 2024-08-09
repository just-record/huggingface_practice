from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(f'enconding: {encoding}')
# enconding: {
# 'input_ids': [101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103, 100, 58263, 13299, 119, 102], 
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
# 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# }

##########################################
### nlptown/bert-base-multilingual-uncased-sentiment
##########################################
# - nlptown: 이 모델을 개발하고 공개한 조직 또는 사용자의 이름
# - bert: 이 모델의 기본 구조가 BERT 아키텍처를 사용
# - base: BERT의 기본 크기 버전 -'small'보다 크고 'large'보다 작은 모델
# - multilingual: 다국어 지원
# - uncased: 대소문자 구분 없음
# - sentiment: 감성 분석 - 긍정/부정


##########################################
### 입력 리스트 가능, padding, truncation로 균일한 길이의 배치를 반환
##########################################
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

print(f'pt_batch: {pt_batch}')
# pt_batch: {
# 'input_ids': tensor([
#                       [  101, 11312, 10320, 12495, 19308, 10114, 11391, 10855, 10103,   100, 58263, 13299,   119,   102],
#                       [  101, 11312, 18763, 10855, 11530,   112,   162, 39487, 10197,   119,    102,     0,     0,     0]
#                     ]), 
# 'token_type_ids': tensor([
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#                     ]), 
# 'attention_mask': tensor([
#                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
#                      ])
# }