from transformers import AutoFeatureExtractor
import librosa
import requests

feature_extractor = AutoFeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")

# 오디오 파일 URL
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

# 오디오 파일 다운로드 및 로드
audio_file = requests.get(audio_url).content
with open("temp_audio.flac", "wb") as f:
    f.write(audio_file)

# librosa를 사용하여 오디오 로드
audio, sample_rate = librosa.load("temp_audio.flac", sr=16000)

# 올바른 입력 형식으로 변환
inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
print(f'inputs: {inputs}')
# inputs: {'input_values': tensor([[ 0.0717,  0.0463, -0.0508,  ..., -0.0046,  0.0326, -0.1615]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1]], dtype=torch.int32)}

