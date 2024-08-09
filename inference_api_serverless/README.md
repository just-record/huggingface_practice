# Inference API Serverless

<https://huggingface.co/docs/api-inference/index>

Hugging Face 공유 인프라에서 호스팅되는 빠른 추론을 통해 간단한 HTTP 요청으로 150,000개 이상의 공개적으로 접근 가능한 머신 러닝 모델 또는 자체 비공개 모델을 무료로 테스트하고 평가할 수 있습니다. 추론 API는 무료로 사용할 수 있으며 속도 제한이 있습니다.

## Overview

  - 01.inference_api_serverless_overview.py

## Detailed parameters

일반적으로 🤗 호스팅 API 추론은 단순한 문자열을 입력으로 받습니다. 하지만 더 고급 사용법은 모델이 해결하는 "작업"에 따라 다릅니다. 모델의 "작업"은 모델 페이지에서 정의됩니다:

<https://huggingface.co/google-bert/bert-base-uncased>: 우측의 Inference API
  
### Natural Language Processing

#### Fill Mask task

  - 02.inference_api_serverless_fill_mask.py
    - 파라미터
      - inputs (필수): 채워질 문자열로, [MASK] 토큰을 포함해야 합니다 (정확한 마스크 이름은 모델 카드에서 확인하세요)
      - options: 다음 키를 포함하는 딕셔너리
        - use_cache (기본값: `true`) 불리언. 
        - wait_for_model (기본값: `false`) 불리언. : 모델이 준비되지 않은 경우, 503 오류를 받는 대신 모델을 기다립니다.
    - 반환값
      - sequence: 모델에 실행된 실제 토큰 시퀀스 (특수 토큰을 포함할 수 있음)
      - score: 이 토큰에 대한 확률
      - token: 토큰의 ID
      - token_str: 토큰의 문자열 표현  

#### Summarization task

  - 03.inference_api_serverless_summarization.py
    - 파라미터
      - inputs (필수): 요약할 문자열
      - parameters: 다음 키를 포함하는 딕셔너리
        - min_length, max_length, ... 생략
    - 반환값
      - summary_text: 요약된 텍스트

#### Question Answering task

- 04.inference_api_serverless_question_answering.py
  - 파라미터
    - inputs (필수): 질문과 문맥을 포함하는 딕셔너리
  - 반환값
    - score: 답변의 확률
    - start: 답변의 시작 인덱스
    - end: 답변의 끝 인덱스
    - answer: 답변

나머지는 생략 - 필요한 task가 있을 때 검색하여 문제 해결