from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
pt_tokenizer = AutoTokenizer.from_pretrained(model_name)

pt_batch = pt_tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

pt_outputs = pt_model(**pt_batch)

from torch import nn

pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(f'pt_predictions: {pt_predictions}')
# pt_predictions: 
# tensor([[0.0021, 0.0018, 0.0115, 0.2121, 0.7725],
#         [0.2084, 0.1826, 0.1969, 0.1755, 0.2365]], grad_fn=<SoftmaxBackward0>)

argmax = pt_predictions.argmax(dim=1)
print(f'argmax: {argmax}')