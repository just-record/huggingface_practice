from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
# ~/.cache/huggingface/hub/models--google--pegasus-xsum/snapshots/8d8ffc158a3bee9fbb03afacdfc347c823c5ec8b/config.json

##########################################
### revision 지정
##########################################
hf_hub_download(
    repo_id="google/pegasus-xsum",
    filename="config.json",
    revision="4d33b01d79672f27f001f6abade33f22d993b151"
)