# Hub Python Libray

<https://huggingface.co/docs/huggingface_hub/index>

`huggingface_hub` 라이브러리를 통해 'Hugging Face Hub'와 상호 작용할 수 있습니다. Hugging Face Hub는 창작자와 협업자를 위한 머신 러닝 플랫폼입니다. 여러분의 프로젝트에 사용할 수 있는 사전 훈련된 모델과 데이터셋을 발견하거나, Hub에서 호스팅되는 수백 개의 머신 러닝 앱을 사용해 볼 수 있습니다. 또한 자신만의 모델과 데이터셋을 만들어 커뮤니티와 공유할 수도 있습니다. `huggingface_hub` 라이브러리는 이 모든 작업을 Python으로 쉽게 수행할 수 있는 방법을 제공합니다.

## Quickstart

### Installation

```bash
pip install --upgrade huggingface_hub
```

### Download files

Hub의 저장소들은 git으로 버전 관리되며, 사용자들은 단일 파일 또는 전체 저장소를 다운로드할 수 있습니다. **hf_hub_download()** 함수를 사용하여 파일을 다운로드할 수 있습니다. 이 함수는 파일을 다운로드하여 로컬 디스크에 캐시합니다. 다음에 그 파일이 필요할 때는 캐시에서 로드하므로 다시 다운로드할 필요가 없습니다.

- 01.hub_python_lib_download_files.py
  - 다운로드된 파일 경로: ~/.cache/huggingface/hub/models--google--pegasus-xsum/snapshots/8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b/config.json

### Authentication

많은 경우에 Hub와 상호 작용하려면 Hugging Face 계정으로 인증을 받아야 합니다: 비공개 저장소 다운로드, 파일 업로드, PR 생성 등... 계정이 없다면 계정을 만들고 로그인하여 설정 페이지에서 사용자 액세스 토큰을 받아야 합니다.

참고로 토큰은 `read` 또는 `write` 권한을 가질 수 있습니다. 저장소를 생성하거나 편집하려면 `write` 액세스 토큰을 가지고 있어야 합니다.

#### Login command

```bash
huggingface-cli login
# Your token has been saved to /home/dev/.cache/huggingface/token
# Login successful
```

- 02.hub_python_lib_authentication.py

나머지는 필요 할 경우 진행
