from transformers import pipeline

##########################################
### 1. task만 설정 하고 모델 지정 없이 사용
##########################################
# pipeline 객체 생성 - task 지정 - automatic-speech-recognition
transcriber = pipeline(task="automatic-speech-recognition")

# pipeline 객체에 음성 파일 경로 전달
response = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': 'I HAVE A DREAM BUT ONE DAY THIS NATION WILL RISE UP LIVE UP THE TRUE MEANING OF ITS TREES'}


##########################################
### 2. task 없이 모델 지정 하기
##########################################
print('-'*30)
transcriber = pipeline(model="openai/whisper-large-v2")
response = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}


##########################################
### 3. 예제가 아닌 다른 모델 지정
# https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending
# openai/whisper-large-v3 => https://huggingface.co/openai/whisper-large-v3
# 우측 상단'Use in transformers' 버튼 클릭 -> 'Transformers' -> 'pipeline'이 있는 코드 복사
##########################################
print('-'*30)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
response = transcriber("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}


##########################################
### 4. 여러 개를 입력하기
##########################################
print('-'*30)
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
response = transcriber(
    [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
    ]
)
print(response)
# [{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}, {'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.'}]