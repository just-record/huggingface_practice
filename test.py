from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
import os

result = login(token=os.environ['HUGGING_FACE_HUB_TOKEN'])
print(result)

# Use a pipeline as a high-level helper
from transformers import pipeline

# pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B", max_length=1000)
pipe = pipeline("text-generation", model="meta-llama/Llama-3.1-8B-Instruct", max_length=1000, truncation=True, device=0)

# response = pipe("AI 분야에서 사용하는 LLM이라는 용어가 뭔지 설명해줘")
response = pipe("""
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
                """)
print(response)