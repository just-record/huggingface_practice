from transformers import pipeline

##########################################
### 1. iterator(yield 사용)를 사용한 Dataset 사용
##########################################
def data():
    # for i in range(1000):
    for i in range(10):
        yield f"My example {i}"


pipe = pipeline(model="openai-community/gpt2", device=0)
generated_characters = 0
for out in pipe(data()):
    print(out[0]["generated_text"])
    generated_characters += len(out[0]["generated_text"])

print(f"Generated {generated_characters} characters in total.")    
# ...
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# My example 8 - to show more than one way to get up from nothing without getting stuck on something by putting a piece of foam on a piece of paper that you're holding in both hands. In my case the paper was going to be at first
# Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
# My example 9 is a simple example. Let's say you want to add an event or some other functionality to your browser, and you want to add the following line to your code:

# event.onclick = function () {

# console
# Generated 1948 characters in total.

##########################################
### 2. Hugging Face의 Datasets 사용
# hf-internal-testing/librispeech_asr_dummy
# 결과: 알 수 없는 결과가 나옴
##########################################
print('-'*30)
# KeyDataset is a util that will just output the item we're interested in.
from transformers.pipelines.pt_utils import KeyDataset
from datasets import load_dataset

pipe = pipeline(model="hf-internal-testing/tiny-random-wav2vec2", device=0)
dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:10]")

for out in pipe(KeyDataset(dataset, "audio")):
    print(out)
# {'text': "EYB  ZB COE C BEZCYCZ HO MOWWB EM BWOB ZMEG  B COEB BE BEC B U OB BE BCB BEWUBB BXYWBESWYCB SBBB SSEZ C Z WH UB F IGVB SB Z<unk> XOES CZ BBXOXFBB  OBY W B VM OFOWUONFWB ZCX B M WZ Q S C Q BC CQBF FOMB BOT ZWYBZ WB  B CM B C B WZCWWW BHU EOYTO YWB BZ SHZBGEM Q OO T B BM XZ QW C OFBZMSEHB BE ZZBX M Q XB<unk> CEVWZ FOHSB W B O Z ZW S ZB O VM <s> D EUCKH XNC D Q BG B O BW U  U  MBE CBYE  WB HFQUBQBUWZ B MW BMPY F ZBU  EB B WBOF S XFOBB ZB X B MOT W B CEO WBM   BBXBBEOBECB B UM C BP FMBWB BZ WFCED Z B B FXB Z OZ OBBZ NVD UBZC W B WYCWY X CE CW B WB MWU BWN B DECF GEF'C WZS CS BYWB<s>FZ'Z<s>ZGBU ECFEY BF ZOZ O UWBSSZBBBBW   O O DBB BZWFUW ZWOZYCGOYCOT WC O CZ BD BBBBBBX X W T B BC BZC FWYBFO FBCE X Z PEZ CE B WEDBMBO BN B BY Y  W B BMCB XOXQ  BSZES Z M CF S FB BBXBB B C CSZ EF SEQF S BEC BNO BN  SU EH  WRFBS WB  W B OEZ WS X B F B X ZBBE BBEHB B BU BECBSXHB BSQWFW BSZXH BWSEG W VQETZMCZ UCXW Z DBE<s> O SXZX MB W RX YYOBSUBWOCFYEF O B O B C Z UBEZBE BTB C   CBFCB V W B BF W ZBBESBBECUT OH YCGZE W BYZBYGBEW BSBE B  B CBEBEBU BU OQB OSB CB Z Z Z HZGTXZ USZFB FC ZO C"}


##########################################
### 3. 다른 Datasets 사용
# PolyAI/minds14
# https://huggingface.co/docs/transformers/quicktour#pipeline
##########################################
print('-'*30)
import torch
from transformers import pipeline
from datasets import load_dataset, Audio

# speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0, trust_remote_code=True)
dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=speech_recognizer.feature_extractor.sampling_rate))
result = speech_recognizer(dataset[:4]["audio"])
print([d["text"] for d in result])
# ['I WOULD LIKE TO SET UP A JOINT ACCOUNT WITH MY PARTNER HOW DO I PROCEED WITH DOING THAT', "FONDERING HOW I'D SET UP A JOIN TO HELL T WITH MY WIFE AND WHERE THE AP MIGHT BE", "I I'D LIKE TOY SET UP A JOINT ACCOUNT WITH MY PARTNER I'M NOT SEEING THE OPTION TO DO IT ON THE APSO I CALLED IN TO GET SOME HELP CAN I JUST DO IT OVER THE PHONE WITH YOU AND GIVE YOU THE INFORMATION OR SHOULD I DO IT IN THE AP AN I'M MISSING SOMETHING UQUETTE HAD PREFERRED TO JUST DO IT OVER THE PHONE OF POSSIBLE THINGS", 'HOW DO I FURN A JOINA COUT']