import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

prompt = """
# 농작업 항목:
'가지유인'
'경운정지'
'김매기'
'물주기'
'병해충방제'
'봉지씌우기'
'봉지벗기기'
'선별'
'포장'
'수확'
'순지르기'
'눈따기'
'열매솎기'
'온도관리'
'운반'
'저장'
'인공수정'
'퇴비주기'
'비료주기'
'하우스관리'
'가지고르기'
'구입'
'판매'
'영농교육'
'비닐씌우기'
'비닐벗기기'
'반사필름'
'잎따기'

* 지시내용: 
농작업을 추출해줘. 작업량이 있는 경우는 작업량도 함께 추출해줘.
반드시 주어진 농작업 항목 중에서 선택 해야 해.
주어진 항목에서 찾을 수 없을 경우는 제외 해줘.
나에게 아주 중요한 작업이야.

# 답볍 예시:
-수확:10Kg
-판매:20Kg
-영농교육


# 요청 문장
오늘 비료를 줬어. 참외는 15키로를 땄고 8키로를 옮기고 7키롤 내다 팔았다.
"""

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    max_length=1000,
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    # {"role": "user", "content": "Who are you?"},
    {"role": "user", "content": prompt},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])
