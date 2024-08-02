# Transformers

<https://huggingface.co/docs/transformers/index>

🤗 Transformers는 최신의 사전 훈련된 모델을 쉽게 다운로드하고 훈련할 수 있는 API와 도구를 제공합니다.

## Tutorials

### Run inference with pipelines

<https://huggingface.co/docs/transformers/pipeline_tutorial>

**pipeline()**은 Hub의 모든 모델을 언어, 컴퓨터 비전, 음성, 그리고 멀티모달 작업에 대한 추론을 쉽게 사용할 수 있게 해줍니다.

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
    - 결과가 이상 함(원인 찾아야 함)
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
  - 원인을 찾아야 함

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
    - 분류 결과가 동일한 모델 검색: 결과가 예상 대로 나오지 않음
     

