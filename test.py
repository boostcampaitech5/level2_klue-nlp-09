import torch
from transformers import AutoTokenizer, BartForSequenceClassification, BartTokenizerFast, PreTrainedTokenizerFast, BartModel

tokenizer = AutoTokenizer.from_pretrained("hyunwoongko/kobart")
model = BartForSequenceClassification.from_pretrained("hyunwoongko/kobart", num_labels=30)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
print(model.config.id2label[predicted_class_id])
exit()
# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BartForSequenceClassification.from_pretrained("hyunwoongko/kobart", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)