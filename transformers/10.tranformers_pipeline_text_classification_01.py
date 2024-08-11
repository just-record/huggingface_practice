from transformers import pipeline

##########################################
### 1. task만 설정 하고 모델 지정 없이 사용
##########################################
pipe = pipeline("text-classification", device=0)
outs = pipe("This restaurant is awesome")
print(outs)
# [{'label': 'POSITIVE', 'score': 0.9998743534088135}]


##########################################
### 2. task 없이 모델 지정 하기
##########################################
print('-'*30)
pipe = pipeline(model="FacebookAI/roberta-large-mnli", device=0)
outs = pipe("This restaurant is awesome")
print(outs)
# [{'label': 'NEUTRAL', 'score': 0.7313143014907837}]
# NEUTRAL(중립), CONTRADICTION(모순), ENTAILMENT(수반) 중 하나로 분류됨 

### 다른 문장으로 테스트 - 2개의 문장: 문장간의 관계를 분류
print('-'*30)
outs = pipe("I like you. I love you")
print(outs)
outs = pipe("I like you. I hate you")
print(outs)
outs = pipe("I like you. I don't hate you")
print(outs)
# [{'label': 'NEUTRAL', 'score': 0.7160009145736694}]
# [{'label': 'CONTRADICTION', 'score': 0.9992231130599976}]
# [{'label': 'ENTAILMENT', 'score': 0.7354484796524048}]


##########################################
### 3. 예제가 아닌 다른 모델 지정
# https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
# BAAI/bge-reranker-v2-m3 => https://huggingface.co/BAAI/bge-reranker-v2-m3
# 우측 상단'Use in transformers' 버튼 클릭 -> 'Transformers' -> 'pipeline'이 있는 코드 복사
##########################################
print('-'*30)
pipe = pipeline("text-classification", model="BAAI/bge-reranker-v2-m3", device=0)
outs = pipe("This restaurant is awesome")
print(outs)
# [{'label': 'LABEL_0', 'score': 0.1711844801902771}]
# LABEL_0(관련성이 낮음), LABEL_1(관련성이 높음) 중 하나로 분류됨

### 다른 문장으로 테스트 - 2개의 문장: 문장간의 관련성을 분류
print('-'*30)
outs = pipe([{"text": "'what is panda?'", "text_pair": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."}])
print(outs)
outs = pipe([{"text": "'what is panda?'", "text_pair": "hi"}])
print(outs)
# [{'label': 'LABEL_0', 'score': 0.9772558808326721}]
# [{'label': 'LABEL_0', 'score': 0.00045623243204317987}]


##########################################
### 4. 텍스트 감정 분석 모델
# https://huggingface.co/models?pipeline_tag=text-classification&sort=trending
# mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis => https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
# 우측 상단'Use in transformers' 버튼 클릭 -> 'Transformers' -> 'pipeline'이 있는 코드 복사
##########################################
print('-'*30)
pipe = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", device=0)
outs = pipe("This restaurant is awesome")
print(outs)
# [{'label': 'neutral', 'score': 0.9986506104469299}]
# positive, neutral, negative 중 하나로
# 결과가 예상 밖인데 => 금융 데이터를 학습한 모델이라서 그런 것 같음
# 모델 카드의 설명: 
#   This model is a fine-tuned version of distilroberta-base on the financial_phrasebank dataset. It achieves the following results on the evaluation set 
#   => 이 모델은 financial_phrasebank 데이터셋에 대해 미세 조정된 distilroberta-base의 버전입니다. 평가 세트에서 다음과 같은 결과를 얻었습니다:

print('-'*30)
outs = pipe("Operating profit totaled EUR 9.4 mn , down from EUR 11.7 mn in 2004 .")
print(outs)
# [{'label': 'negative', 'score': 0.9987391829490662}]