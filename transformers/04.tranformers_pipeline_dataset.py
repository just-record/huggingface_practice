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
# My example 8-bit console runs on Windows (x86_64). I also have 2-bit 64 bit OS's I use on my Mac (x86_64)!
# My example 9th grade teacher works in a church. He's a big fan of Harry Potter, and he knows he's right. He's not going to let someone else decide who he feels like to make his life harder. He knows one girl
# Generated 2013 characters in total.

##########################################
### 2. Hugging Face의 Datasets 사용
# hf-internal-testing/librispeech_asr_dummy
# 결과: 결과가 text로 나오지 않음
#   hf-internal-testing/librispeech_asr_dummy이 모델 목록에서 검색 되지 않음
#   hf-internal-testing 에서 test용 모델 인 듯
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
### 앞 코드의 모델로 변경 
##########################################
print('-'*30)
pipe = pipeline(model="openai/whisper-large-v2", device=0)
# {'text': ' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.'}
# {'text': " Nor is Mr. Quilter's manner less interesting than his matter."}
# {'text': ' He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind.'}
# {'text': " He has grave doubts whether Sir Frederick Leighton's work is really Greek after all, and can discover in it but little of Rocky Ithaca."}
# {'text': " Linnell's pictures are a sort of up-guards-and-addams paintings, and Mason's exquisite idylls are as national as a jingo poem. Mr. Burkett Foster's landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And Mr. John Collier gives his sitter a cheerful slap on the back, before he says, like a shampoo-er in a Turkish bath,"}
# {'text': ' It is obviously unnecessary for us to point out how luminous these criticisms are, how delicate in expression.'}
# {'text': ' On the general principles of art, Mr. Quilter writes with equal lucidity.'}
# {'text': ' Painting, he tells us, is of a different quality to mathematics, and Finnish in art is adding more factor.'}
# {'text': ' As for etchings, they are of two kinds, British and foreign.'}
# {'text': ' He laments most bitterly the divorce that has been made between decorative art and what we usually call pictures, makes a customary appeal to the last judgment, and reminds us that in the great days of art Michelangelo was the furnishing upholsterer.'}


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