# Transformers

<https://huggingface.co/docs/transformers/index>

🤗 Transformers는 최신의 사전 훈련된 모델을 쉽게 다운로드하고 훈련할 수 있는 API와 도구를 제공합니다.

## Tutorials

### Run inference with pipelines

<https://huggingface.co/docs/transformers/pipeline_tutorial>

pipeline()은 Hub의 모든 모델을 언어, 컴퓨터 비전, 음성, 그리고 멀티모달 작업에 대한 추론을 쉽게 사용할 수 있게 해줍니다.

#### Pipeline usage

<https://huggingface.co/docs/transformers/pipeline_tutorial#pipeline-usage>

'pipeline()'은 자동으로 기본 모델과 작업에 대한 추론이 가능한 전처리 클래스를 로드합니다. 

- 01.tranformers_pipeline_basic.py: 자동 음성 인식(ASR) 또는 음성-텍스트 변환 예시
  - 1. task만 설정 하고 모델 지정 없이 사용
  - 2. task 없이 모델 지정 하기: 'openai/whisper-large-v2'
  - 3. 예제가 아닌 다른 모델 지정: 'openai/whisper-large-v3'

```bash
# ValueError: ffmpeg was not found but is required to load audio files from filename
# 위와 같은 에러가 발생하면 ffmpeg를 설치해야 합니다.
sudo apt update
sudo apt install ffmpeg
```

##### 지원 되는 tasks

소스 코드를 참조하면 다음과 같은 task들을 지원하는 걸 확인할 수 있습니다.

<https://github.com/huggingface/transformers/blob/v4.43.3/src/transformers/pipelines/__init__.py#L552>

- `"audio-classification"`: will return a [`AudioClassificationPipeline`].
- `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
- `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
- `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
- `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
- `"fill-mask"`: will return a [`FillMaskPipeline`]:.
- `"image-classification"`: will return a [`ImageClassificationPipeline`].
- `"image-feature-extraction"`: will return an [`ImageFeatureExtractionPipeline`].
- `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
- `"image-to-image"`: will return a [`ImageToImagePipeline`].
- `"image-to-text"`: will return a [`ImageToTextPipeline`].
- `"mask-generation"`: will return a [`MaskGenerationPipeline`].
- `"object-detection"`: will return a [`ObjectDetectionPipeline`].
- `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
- `"summarization"`: will return a [`SummarizationPipeline`].
- `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
- `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
- `"text-classification"` (alias `"sentiment-analysis"` available): will return a [`TextClassificationPipeline`].
- `"text-generation"`: will return a [`TextGenerationPipeline`]:.
- `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
- `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
- `"translation"`: will return a [`TranslationPipeline`].
- `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
- `"video-classification"`: will return a [`VideoClassificationPipeline`].
- `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
- `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
- `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
- `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
- `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].

#### Parameters

<https://huggingface.co/docs/transformers/pipeline_tutorial#parameters>

'pipeline()'은 많은 매개변수를 지원합니다. 일부는 특정 작업에 관련되고, 일부는 모든 파이프라인에 공통적입니다. 일반적으로 원하는 곳 어디에서나 매개변수를 지정할 수 있습니다.

> Device, batch_size

- 02.tranformers_pipeline_params_device.py: pipeline 매개변수(device) 설정 예시
  - 1. 'device' - '0': GPU, '-1': CPU
  - 2. 'device_map' - 모델 가중치를 어떻게 로드하고 저장할지 자동으로 결정
  - 3. 'batch_size' - 2로 설정 => 오류 발생(원인 찾아야 함), 1로 설정 시 작동
    - 기본적으로 pipeline은 배치 추론을 하지 않지만 사용 할 경우의 예시

- 03.tranformers_pipeline_params_task_spectific.py: task에 따른 매개변수 설정 예시
  - 1. 'return_timestamps' - 발음된 시간
  - 2. 'chunk_length_s' - 긴 오디오 파일을 작은 청크(chunk)로 나누며 각 청크의 길이를 초 단위로 지정

#### Using pipelines on a dataset

<https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipelines-on-a-dataset>

pipeline은 대규모 데이터셋에 대해 추론을 실행할 수 있으며 권장되어 지는 방법은 반복자를 사용하는 것입니다.

- 04.tranformers_pipeline_dataset.py: Dataset 사용 예시
  - 1. iterator(yield 사용)를 사용한 Dataset 사용
  - 2. Hugging Face의 Datasets 사용 - hf-internal-testing/librispeech_asr_dummy
    - 결과: 결과가 text로 나오지 않음
      - hf-internal-testing/librispeech_asr_dummy이 모델 목록에서 검색 되지 않음
      - hf-internal-testing 에서 test용 모델 인 듯    
    - 앞 코드의 모델로 변경(openai/whisper-large-v2) -> 정상 작동
  - 3. 다른 Datasets 사용 - PolyAI/minds14
    - https://huggingface.co/docs/transformers/quicktour#pipeline

#### Using pipelines for a webserver

<https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipelines-for-a-webserver>

pass

#### Vision pipeline

<https://huggingface.co/docs/transformers/pipeline_tutorial#vision-pipeline>

Vision 작업에 'pipeline()'을 사용하는 것은 실제로 거의 동일

- 05.tranformers_pipeline_vision.py: Vision 작업의 pipeline 사용 예시

#### Text pipeline

<https://huggingface.co/docs/transformers/pipeline_tutorial#text-pipeline>

- 06.tranformers_pipeline_text.py: Text 작업의 pipeline 사용 예시

#### Multimodal pipeline

<https://huggingface.co/docs/transformers/pipeline_tutorial#multimodal-pipeline>

'pipeline()'은 여러 모달리티(modality)를 지원합니다. 이미지와 이미지에 대한 질문을 입력 으로 받는 pipeline을 사용하는 방법입니다.

- 07.tranformers_pipeline_multimodal.py: Multimodal 작업의 pipeline 사용 예시
  - 오류 발생: AttributeError: 'list' object has no attribute 'numpy'
  - 원인을 찾아야 함 => pip install --upgrade transformers => 해결 됨

```bash
# 설치 필요
sudo apt install -y tesseract-ocr
pip install pytesseract
```

#### Using pipeline on large models with 🤗 accelerate

<https://huggingface.co/docs/transformers/pipeline_tutorial#using-pipeline-on-large-models-with--accelerate->

Hugging Face의 accelerate 라이브러리를 사용하면 대규모 모델을 쉽게 실행할 수 있습니다.

- 08.tranformers_pipeline_accelerate.py: accelerate 라이브러리 사용 예시
  - 1.3B 파라미터 버전의 텍스트 생성 모델

#### Creating web demos from pipelines with gradio

<https://huggingface.co/docs/transformers/pipeline_tutorial#creating-web-demos-from-pipelines-with-gradio>

- 09.tranformers_pipeline_gradio.py: gradio를 이용한 웹데코 생성 예시
  - <http://127.0.0.1:7860/>

#### 추가 연습

위의 tutorial을 참고하여 추가 연습을 진행합니다.

##### text-classification

<https://huggingface.co/docs/transformers/main_classes/pipelines>

- 10.tranformers_pipeline_text_classification_01.py: Text Classification 예시
  - 1. task만 설정 하고 모델 지정 없이 사용
  - 2. task 없이 모델 지정 하기: 'FacebookAI/roberta-large-mnli'
    - 분류 범주가 다른 모델: 예제가 아닌 다른 문장 입력
  - 3. 예제가 아닌 다른 모델 지정: 'BAAI/bge-reranker-v2-m3'
    - 분류 범주가 다른 모델: 예제가 아닌 다른 문장 입력
  - 4. 텍스트 감정 분석 모델: 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    - 분류 결과가 동일한 모델 검색: 결과가 예상 대로 나오지 않음(금융 데이터를 학습 한 모델) -> 문장 변경
     

##### text-generation

- 11.transformers_pipeline_text_generation_01.py: 최신 모델 사용
  - meta-llama/Meta-Llama-3.1-8B-Instruct: 모델의 접근 시 인증 필요
    - 입력 항목: 'First Name', 'Last Name', 'Data of birth', 'Country', 'Affitiation', 'Job title'
    - '라이선스 동의', '개인정보 처리 동의'
    - 모델 소유자의 승인 필요
  - Hugging Face의 Access Tokens
    - 'Profile'(우측 상단) -> 'Settings' -><'Access Tokens' -> '+Create new token'
    - Token type: Write, Token name: xxx -> 'Create token' -> Copy
  - Hugging Face login 코드 추가 - Access Token 필요
  - CUDA out of memory
    - 우선 실행 시 시간이 너무 오래 걸림. 
    - torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 224.00 MiB. GPU 0 has a total capacity of 23.68 GiB of which 119.06 MiB is free. Including non-PyTorch memory, this process has 23.49 GiB memory in use. Of the allocated memory 23.24 GiB is allocated by PyTorch, and 1.17 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

### Write portable code with AutoClass

<https://huggingface.co/docs/transformers/autoclass_tutorial>

AutoClass는 주어진 체크포인트에서 올바른 아키텍처를 자동으로 추론하고 로드합니다. from_pretrained() 메소드를 사용하면 어떤 아키텍처의 사전 훈련된 모델도 빠르게 로드할 수 있어, 모델을 처음부터 훈련하는 데 시간과 리소스를 들일 필요가 없습니다. 이러한 체크포인트에 구애받지 않는 코드를 작성하면, 한 체크포인트에서 작동하는 코드는 아키텍처가 다르더라도 유사한 작업을 위해 훈련된 다른 체크포인트에서도 작동할 것입니다.

- Architecture (아키텍처): 아키텍처는 모델의 구조나 "뼈대"를 의미
  - 예를 들어, BERT, GPT, ResNet 등이 각각 다른 아키텍처입니다.
- Checkpoint (체크포인트): 특정 아키텍처에 대해 학습된 가중치(weights)의 집합
  - 예를 들어, 'google-bert/bert-base-uncased'는 BERT 아키텍처의 한 체크포인트
- Model (모델): 아키텍처나 체크포인트를 모두 지칭할 수 있어 문맥에 따라 모델이 아키텍처를 의미할 수도 있고, 특정 체크포인트를 의미할 수도 있음
  - 예를 들어, "BERT 모델"이라고 할 때는 BERT 아키텍처를 의미할 수 있지만, "사전 훈련된 BERT 모델을 로드했다"라고 하면 특정 체크포인트를 의미할 가능성이 높습니다.

#### AutoTokenizer

거의 모든 NLP 작업은 토크나이저로 시작합니다. 토크나이저는 입력을 모델이 처리할 수 있는 형식으로 변환합니다.

- 12.transformers_autoclass_autotokenizer.py: 'AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")'

#### AutoImageProcessor

비전 작업의 경우, 이미지 프로세서는 이미지를 올바른 입력 형식으로 처리합니다.

- 13.transformers_autoclass_autoimageprocessor.py: 'AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")'

#### AutoBackbone

AutoBackbone은 사전 훈련된 모델을 백본으로 사용하여 백본의 다양한 단계에서 특징 맵을 얻을 수 있게 해줍니다.

- 14.transformers_autoclass_autobackbone.py: 'AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))'
  - 'out_indices': 특징 맵을 얻고자 하는 레이어의 인덱스
  - 'out_features': 특징 맵을 얻고자 하는 레이어의 이름

#### AutoFeatureExtractor

오디오 작업의 경우, 특성 추출기가 오디오 신호를 올바른 입력 형식으로 처리합니다.

- 15.transformers_autoclass_autofeatureextractor.py

#### AutoModel

Pytorch: AutoModelFor 클래스를 사용하면 주어진 작업에 대해 사전 훈련된 모델을 로드할 수 있습니다. 예제만 봐선 이해가 어렵습니다.

- 16.transformers_autoclass_automodelfor.py 
  - DistilBERT는 BERT(Bidirectional Encoder Representations from Transformers)의 경량화 버전입니다. 원래 BERT 모델의 성능을 거의 유지하면서도 크기와 속도 면에서 개선된 모델입니다.
  - "base": 기본 크기의 모델을 의미합니다 (대형 모델도 있습니다).
  - "uncased": 텍스트를 소문자로 변환하여 대소문자를 구분하지 않는 모델임을 나타냅니다.

##### Quick tour 중 일부

<https://huggingface.co/docs/transformers/quicktour#autoclass>

AutoModelForSequenceClassification과 AutoTokenizer 클래스는 내부적으로 함께 작동하여 위에서 사용한 **pipeline()**을 구동합니다. AutoClass는 사전 훈련된 모델의 아키텍처를 그 이름이나 경로로부터 자동으로 불러오는 단축 방법입니다. 여러분은 단지 작업에 적합한 AutoClass와 그에 연관된 전처리 클래스를 선택하기만 하면 됩니다.

> AutoTokenizer

토크나이저는 텍스트를 모델의 입력으로 사용할 숫자 배열로 전처리하는 역할을 합니다. 토큰화 과정을 관리하는 여러 규칙이 있으며, 여기에는 단어를 어떻게 분할할지, 어느 수준에서 단어를 분할해야 할지 등이 포함됩니다. 가장 중요한 것은 모델이 사전 훈련된 것과 동일한 토큰화 규칙을 사용하고 있음을 보장하기 위해 같은 모델 이름으로 토크나이저를 인스턴스화해야 한다는 점입니다.

토크나이저가 반환하는 딕셔너리에는 다음이 포함됩니다:
- input_ids: 토큰의 수치적 표현입니다.
- attention_mask: 어떤 토큰에 주의를 기울여야 하는지를 나타냅니다.

- 17.transformers_autoclass_autotokenizer.py
  - nlptown/bert-base-multilingual-uncased-sentiment
    - nlptown: 이 모델을 개발하고 공개한 조직 또는 사용자의 이름
    - bert: 이 모델의 기본 구조가 BERT 아키텍처를 사용
    - base: BERT의 기본 크기 버전 -'small'보다 크고 'large'보다 작은 모델
    - multilingual: 다국어 지원
    - uncased: 대소문자 구분 없음
    - sentiment: 감성 분석 - 긍정/부정

> AutoModel

Transformers는 사전 훈련된 인스턴스를 로드하는 간단하고 통일된 방법을 제공합니다. 이는 AutoTokenizer를 로드하는 것과 같은 방식으로 AutoModel을 로드할 수 있다는 것을 의미합니다. 유일한 차이점은 작업에 맞는 올바른 AutoModel을 선택하는 것입니다. 텍스트(또는 시퀀스) 분류의 경우, AutoModelForSequenceClassification을 로드해야 합니다.

- 18.transformers_autoclass_automodelfor.py

### Preprocess data

<https://huggingface.co/docs/transformers/preprocessing#natural-language-processing>

- 모델에서 데이터셋을 학습하기 전에, 모델 입력 형식으로 전처리가 필요
- 🤗 Transformers는 데이터를 모델에 맞게 준비하는 데 도움이 되는 전처리 클래스 세트를 제공

✔️ 데이터 유형별 전처리 도구

- 텍스트: Tokenizer
- 음성/오디오: Feature extractor
- 이미지: ImageProcessor
- 복합데이터: Processor

AutoProcessor: 모델에 맞는 전처리 도구를 자동으로 선택. 어떤 데이터 유형이든 알아서 처리.

✔️ 설치

```bash
pip install datasets
```

#### Natural Language Processing

토크나이저: 텍스트 데이터 전처리의 핵심 도구

1. 역할
- 텍스트를 규칙에 따라 *토큰*으로 분리
- 토큰을 숫자로 변환
- 숫자를 텐서로 변환 (모델 입력용)
- 모델에 필요한 추가 입력도 자동 생성

2. 사전 학습 모델 사용시 주의사항
- 반드시 해당 모델의 사전 학습된 토크나이저를 사용해야 함
- 이유: 사전 학습 때와 동일한 방식으로 텍스트 분리 필요
- 동일한 토큰-인덱스 매핑(*vocab*) 사용 보장

- 19.transformers_preprocess_nlp.py
  - tokenizer: 숫자로 변환
  - tokenizer.decode: 토큰을 다시 텍스트로 변환
  - batch: 여러 문장을 한 번에 처리

✔️ Pad

1. 문제점:
- 문장들의 길이가 서로 다름
- 모델 입력(텐서)은 균일한 크기가 필요
- 길이가 다르면 텐서로 변환 불가

2. 해결방법: 패딩(Padding)
- 짧은 문장에 특별한 *패딩 토큰*을 추가
- 가장 긴 문장 길이에 맞춰 나머지 문장들을 채움

- 20.transformers_preprocess_nlp_pad.py

✔️ Truncation

1. 문제점:
- 문장이 모델이 처리할 수 있는 길이보다 긴 경우
- 모델마다 처리할 수 있는 최대 길이가 정해져 있음

2. 해결방법: 잘라내기(Truncation)
- 긴 문장을 모델의 최대 허용 길이에 맞게 자름

- 21.transformers_preprocess_nlp_truncation.py

✔️ Build tensors

1. 목적:
- 모델 입력용 텐서 생성
- PyTorch 또는 TensorFlow 형식으로 변환

- 22.transformers_preprocess_nlp_build_tensors.py

#### Audio, Computer Vision, Multimodal

추후 필요 시 진행