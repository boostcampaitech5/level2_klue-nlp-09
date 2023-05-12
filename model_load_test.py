from transformers import AutoModelForSequenceClassification, T5TokenizerFast, AutoTokenizer, PreTrainedTokenizerFast, T5ForConditionalGeneration, BartModel, BertTokenizerFast, AutoModel
from torch import tensor

def test(model_name):
    input_test1, input_test2 = "테스트용 문장 생성", "테스트용 문장 생성 2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_test = tokenizer(input_test1, input_test2, return_tensors="pt", padding=True, truncation=True, max_length=20, add_special_tokens=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(input_test.keys())
    print(model(input_test['input_ids'])['logits'])

    
# models = ["kykim/electra-kor-base", "xlm-roberta-large", "beomi/KcELECTRA-base", "monologg/koelectra-base-v3-discriminator"]
model = "google/rembert"
test(model)
