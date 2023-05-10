from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerFast, BartForSequenceClassification, BartModel, BertTokenizerFast

def test(model_name):
    input_test1, input_test2 = "테스트용 문장 생성", "테스트용 문장 생성 2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_test = tokenizer(input_test1, input_test2, return_tensors="pt", padding=True, truncation=True, max_length=20, add_special_tokens=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print(model(input_test['input_ids'])['logits'])

model = "hyunwoongko/kobart"
test(model)
