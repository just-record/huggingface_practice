from transformers import pipeline

##########################################
### 1. device=0 사용
# device=0: 첫번째 GPU 사용 - GPU가 없으면 오류 발생
# device=-1: CPU 사용
##########################################
pipe = pipeline(model="openai/whisper-large-v2", device=0)
# pipe = pipeline(model="openai/whisper-large-v2", device=-1)
response = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}


##########################################
### 2. device_map="auto" 사용 - 모델 가중치를 어떻게 로드하고 저장할지 자동으로 결정
##########################################
print('-'*30)
# pipe = pipeline(model="openai/whisper-large-v2", device_map="auto", low_cpu_mem_usage=True)
pipe = pipeline(model="openai/whisper-large-v2", device_map="auto")
response = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}

##########################################
### 3. batch_size 사용 - 한 번에 처리할 음성 파일의 개수
# batch_size=2: 오류 발생 - IndexError: tuple index out of range => 원인을 찾아야 함
# batch_size=1: 정상 처리
##########################################
print('-'*30)
# pipe = pipeline(model="openai/whisper-large-v2", device=0, batch_size=2)
pipe = pipeline(model="openai/whisper-large-v2", device=0, batch_size=1)
audio_filenames = [f"https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/{i}.flac" for i in range(1, 5)]
texts = pipe(audio_filenames)
print(texts)
# [{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.'}, {'text': ' Stuff it into you, his belly counselled him.'}, {'text': ' After early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels.'}, {'text': ' Y en las ramas medio sumergidas revoloteaban algunos pájaros de quimérico y legendario plumaje.'}]