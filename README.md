# 문장 내 개체간 관계 추출
![asdf](https://user-images.githubusercontent.com/82187742/236622385-1af75b87-b5ef-4028-9b82-a52981007cf7.png)



## 파일 구조
```
level2_klue-nlp-09
|-- README.md
|-- best_model
|-- config
|   `-- config.yaml
|-- eda
|   |-- JYS.ipynb
|   |-- KSH.ipynb
|   |-- LDH.ipynb
|   `-- LJS.ipynb
|-- inference.py
|-- load_data.py
|-- prediction
|-- requirements.txt
|-- results
|-- train.py
|-- .gitgnore
`-- utils.py
```

## multi stage model 사용 방법
# 모델: klue_bert_base
1. klue_bert_base_binary.yaml 파일 다음과 같이 수정
- # data
train_path: ../dataset/train/binary_train_90.csv
dev_path: ../dataset/dev/binary_dev_10.csv
predict_path: ../dataset/test/test_data.csv
- # inference
inference_model_name: 사용하고자 하는 binary 분류에 대한 모델

2. klue_bert_base_PER.yaml 파일 다음과 같이 수정
-  data
train_path: ../dataset/train/binary_train_per_90.csv
dev_path: ../dataset/dev/binary_dev_per_10.csv
predict_path: ../dataset/test/binary_test_per.csv
-  inference
inference_model_name: 사용하고자 하는 per 분류에 대한 모델

3. klue_bert_base_ORG.yaml 파일 다음과 같이 수정
-  data
train_path: ../dataset/train/binary_train_org_90.csv
dev_path: ../dataset/dev/binary_dev_org_10.csv
predict_path: ../dataset/test/binary_test_per.csv
-  inference
inference_model_name: 사용하고자 하는 org 분류에 대한 모델

4. 학습
1. 모델 A 학습(binary 분류)
python3 binary_train.py
2. 모델 C(per 분류), D(org 분류) 학습
python3 train.py
3. inference
python3 binary_inference.py