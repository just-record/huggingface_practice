from transformers import pipeline

##########################################
### 'return_timestamps' - 발음된 시간
##########################################
pipe = pipeline(model="openai/whisper-large-v2", device=0, return_timestamps=True)
response = pipe("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(response)
# {'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.', 'chunks': [{'timestamp': (0.0, 11.88), 'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its'}, {'timestamp': (11.88, 12.38), 'text': ' creed.'}]}


##########################################
### 'chunk_length_s' - 긴 오디오 파일을 작은 청크(chunk)로 나누며 각 청크의 길이를 초 단위로 지정
##########################################
print('-'*30)
pipe = pipeline(model="openai/whisper-large-v2", device=0, chunk_length_s=30)
response = pipe("https://huggingface.co/datasets/reach-vb/random-audios/resolve/main/ted_60.wav")
print(response)
# {'text': " So in college, I was a government major, which means I had to write a lot of papers. Now, when a normal student writes a paper, they might spread the work out a little like this. So, you know. You get started maybe a little slowly, but you get enough done in the first week that with some heavier days later on, everything gets done and things stay civil. And I would want to do that like that. That would be the plan. I would have it all ready to go, but then actually the paper would come along, and then I would kind of do this. And that would happen every single paper. But then came my 90-page senior thesis, a paper you're supposed to spend a year on. I knew for a paper like that, my normal workflow was not an option, it was way too big a project. So I planned things out and I decided I kind of had to go something like this. This is how the year would go. So I'd start off light and I'd bump it up"}