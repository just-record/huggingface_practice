from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

##########################################
### 1. model로 감정 분류 하기
##########################################
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

text = "This restaurant is awesome"

inputs = tokenizer(text, return_tensors="pt")
print(f'inputs: {inputs}')
# inputs: {
# 'input_ids': tensor([[  101,  2023,  4825,  2003, 12476,   102]]), 
# 'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
# }
# - input_ids: 토큰의 수치적 표현입니다.
# - attention_mask: 어떤 토큰에 주의를 기울여야 하는지를 나타냅니다.

outputs = model(**inputs)
print(f'outputs: {outputs}')
# outputs: SequenceClassifierOutput(loss=None, logits=tensor([[-0.0002, -0.0701]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
print(f'outputs logits: {outputs.logits[0]}')
# outputs logits: tensor([-0.0002, -0.0701], grad_fn=<SelectBackward0>)

##########################################
### distilbert/distilbert-base-uncased
##########################################
# DistilBERT는 BERT(Bidirectional Encoder Representations from Transformers)의 경량화 버전입니다. 원래 BERT 모델의 성능을 거의 유지하면서도 크기와 속도 면에서 개선된 모델입니다.
# "base": 기본 크기의 모델을 의미합니다 (대형 모델도 있습니다).
# "uncased": 텍스트를 소문자로 변환하여 대소문자를 구분하지 않는 모델임을 나타냅니다.
