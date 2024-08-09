from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

print(image_processor)
# Fast image processor class <class 'transformers.models.vit.image_processing_vit_fast.ViTImageProcessorFast'> is available for this model. Using slow image processor class. To use the fast image processor class set `use_fast=True`.
# ViTImageProcessor {
#   "do_normalize": true,
#   "do_rescale": true,
#   "do_resize": true,
#   "image_mean": [
#     0.5,
#     0.5,
#     0.5
#   ],
#   "image_processor_type": "ViTImageProcessor",
#   "image_std": [
#     0.5,
#     0.5,
#     0.5
#   ],
#   "resample": 2,
#   "rescale_factor": 0.00392156862745098,
#   "size": {
#     "height": 224,
#     "width": 224
#   }
# }

print('-'*30)

import requests
from PIL import Image
import io

image_url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(image_url)
image = Image.open(io.BytesIO(response.content))

# 이미지 전처리
inputs = image_processor(images=image, return_tensors="pt")

print("전처리된 이미지 텐서 shape:", inputs.pixel_values.shape)
print("입력 이미지의 타입:", type(inputs))

# 전처리된 이미지의 주요 속성 출력
for key, value in inputs.items():
    print(f"{key}: {type(value)} - shape: {value.shape}")