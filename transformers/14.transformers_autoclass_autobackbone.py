from transformers import AutoImageProcessor, AutoBackbone
import torch
from PIL import Image
import requests


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = AutoBackbone.from_pretrained("microsoft/swin-tiny-patch4-window7-224", out_indices=(1,))

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
feature_maps = outputs.feature_maps

print(feature_maps[0].shape)
# torch.Size([1, 96, 56, 56])
# 첫 번째 차원 (1): 배치 크기
# 두 번째 차원 (96): 채널 수 또는 특성 맵(feature map)의 수
# 세 번째 차원 (56): 높이
# 네 번째 차원 (56): 너비