# 문장 내 개체간 관계 추출

![asdf](https://user-images.githubusercontent.com/82187742/236622385-1af75b87-b5ef-4028-9b82-a52981007cf7.png)

---

# 1. 개요

### 대회 Task: **문장 내 개체간 관계 추출**

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 모델이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.

### 활용 장비 및 재료

- **서버**: AI Stage (NVIDIA V100 32GB)
- **IDE**: VSCode, Jupyter Lab

- **협업**: Git(GitHub), Notion, Slack
- **모니터링**: WandB

---

# 2. 팀 구성 및 역할

### 김세형\_T5038

- 데이터 EDA 및 preprocessing(easy data aug.)
- Hyperparameter tuning: kogpt2, twhin-bert-large, xlm-roberta-large
- Model evaluation(출력 결과 분석 등)

### 이준선\_T5157

- pytorch lightning base code 작성
- 데이터 EDA
- Hyperparameter tuning: klue/bert-base, klue/roberta-large

### 홍찬우\_T5227

- 데이터 EDA (단어 빈도 분석, [UNK] 토큰 확인)
- 모델 탐색 및 성능 비교
- Hyperparameter tuning : mluke, kobart

### 이동호\_T5139

- 데이터 EDA
- Hyperparameter tuning: google/rembert, klue/bert-base
- 데이터 preprocessing(Entity Representation)

### 정윤석\_T5194

- 데이터 Preprocessing 연결 파일
- Clean Foreign Language 함수 작성
- snunlp/kr-electra-discriminator Modeling
- 데이터 EDA

---

# 3. 수행 절차 및 방법

## 3.0. Base Code 작성(이준선)

- **pytorch lightning 기반 base code 작성**
  - 실험의 편의성 향상(logging, sweep 등)
  - Dataloader 커스텀 용이(토큰 추가, 전처리 등)

## 3.1. EDA

### 3.1.1. Data Distribution (김세형)

- 기본적으로 데이터는 `id`, `sentence`, `subject_entity`, `object_entity`, `label`, `source`로 구성
- `train` 데이터의 경우 32,470개, `test` 데이터의 경우 7,765개가 존재
- 크게 `no_relation`, `per`(person), `org`(organization)의 세 main-label, 그 아래로 `per` 이하 17개, `org` 이하 12개의 sub-label이 존재하여 총 30개의 class로 분류

**Sub-label 별 data distribution [그림 3.1]**

![그림 3.1. Sub-label 별 data distribution](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/1.png)

그림 3.1. Sub-label 별 data distribution

- 가장 기본적으로 30개의 sub-label 별로 몇 개의 데이터가 분포하는지 확인
- `no_relation` 데이터는 32,470개 중 9,534개(약 29.36%)를 차지하며 가장 많은 비율을 보였으며, 이를 제외한 데이터 중 가장 많은 데이터는 `org:top_member/employee`(4,284개, 13.19%), 가장 적은 데이터는 `per:place_of_death`(40개, 0.12%)로 확인되어, imbalance가 매우 큰 것으로 조사되었음

**Main-label 별 data distribution [그림 3.2]**

![그림 3.2. Main-label 별 data distribution](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/2.png)

그림 3.2. Main-label 별 data distribution

- Sub-label 별 distribution 확인 결과 main-label 별 데이터 비율은 sub-label 만큼 상이하지 않을 것이라 판단하였고, 추후 main-label 분류 → sub-label 분류 형태의 모델 제작의 가능성을 조사하기 위해 3개의 main-label 별 분포 확인
- `no_relation`(NR)과 `per` 데이터는 각 9,534개(29.36%), 9,081개(27.97%)로 유사하며, org 데이터가 13,855개(42.67%)로 가장 많음을 확인
- Main-label 내 sub-label 데이터 분포 확인 [그림 3.3, 3.4]
  ![그림 3.3. `org` label 내 data distribution](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/3.png)
  그림 3.3. `org` label 내 data distribution
  ![그림 3.4. `per` label 내 data distribution](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/4.png)
  그림 3.4. `per` label 내 data distribution
  - 다만, sub-label이 없어 가장 많은 sub-label 비율을 차지한 no_relation 데이터를 제외하더라도, org 데이터와 per 데이터의 imbalance는 높으므로 분류 과정을 분할한다 하더라도 여전히 handling이 필요할 것으로 확인되었음

**Source 별 data distribution**

- 전체 데이터는 `wikipedia`, `wikitree`, `policy-briefing`의 세 개 source로부터 확보되었으며, 해당 데이터의 추후 모델 학습에 대한 활용 가능성 조사를 위해 source 별로 데이터 분포를 확인
- train 데이터[그림 3.5]의 경우 `wikipedia` 데이터의 비율이 60% 이상으로 높고, `policy-briefing` 데이터의 경우 1% 미만의 극소수를 차지하는 것으로 확인
  ![그림 3.5. Train 데이터의 source 분포](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/5.png)
  그림 3.5. Train 데이터의 source 분포
  - train 데이터의 source 중 극소수인 `policy-briefing`을 제외한 `wikipedia`, `wikitree`의 데이터 sub-label 분포 확인
    ![그림 3.6. Wikipedia source 데이터 내 sub-label 분포](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/6.png)
    그림 3.6. Wikipedia source 데이터 내 sub-label 분포
    ![그림 3.7. Wikitree source 데이터 내 sub-label 분포](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/7.png)
    그림 3.7. Wikitree source 데이터 내 sub-label 분포
    - 분석 결과, `wikipedia` 데이터에 전체 9,534개 `no_relation` 데이터 중 7,382개 데이터가 존재함을 확인하였고, `wikitree`의 경우 `org:top_member/employee` 데이터 대다수가 해당 소스에서 출처한 모습을 확인
    - 결론적으로, source 별로 데이터 분포의 차이가 명확히 존재함을 보았음
- 허나 test 데이터[그림 3.8]의 경우에 `wikitree` 비율이 과반 이상 높은 모습을 확인할 수 있어, test 데이터의 sub-label 분포가 train 데이터와는 다를 수도 있다는 가능성을 확인
  ![그림 3.8. Test 데이터의 source 분포](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/8.png)
  그림 3.8. Test 데이터의 source 분포

**Token sequence length distribution [그림 3.9, 3.10]**

![그림 3.9. train 데이터 token sequence length(BERT tokenizer)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/9.png)

그림 3.9. train 데이터 token sequence length(BERT tokenizer)

![그림 3.10. test 데이터 token sequence length(BERT tokenizer)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/10.png)

그림 3.10. test 데이터 token sequence length(BERT tokenizer)

- 베이스라인 코드 모델인 `klue/bert-base` 모델의 BERT tokenizer를 이용해 토큰화된 token sequence length를 확인한 결과, train 데이터와 test 데이터의 분포에는 큰 차이가 없음을 확인

### 3.1.2. `[UNK]` Tokens (정윤석)

**unk 토큰으로 바뀌는 단어들을 조사**

```python
{"'": 337, '李': 225, '崔': 60, '皇': 54, '后': 48, '–': 43, '.': 42, '尹': 38, '永': 31, '昌': 31,
'慶': 28, '宋': 28, '趙': 25, '홋스퍼': 24, '興': 23, ')': 22, '盧': 22, '承': 22, '梁': 22, '孝': 21,
 '徐': 21, '姜': 21, '!': 21, '沈': 20, '容': 19, '陵': 19, '申': 18, '放': 18, '池': 18, '貞': 18,
'洪': 18, '鍾': 18, '妃': 16, '俊': 16, '泰': 16, '吳': 16, '進': 15, '洙,': 15, '校': 15,
'홋카이도': 15, '炳': 15, '康': 14, '柳': 14, '唐': 14, '崇': 14, '을': 14, '少': 13, '劉': 13,
'景': 13, '俊,': 13, '團': 13, '☎031': 13, '賢': 13, '忠': 13, '恩': 13, '夏': 12, '嬪': 12,
 '惠': 12, '煥,': 12, '翁': 12, '숀': 12, '錫': 11, '羅': 11, '熙,': 11, '許': 11, '植,': 10,
'樂': 10, '延': 10, '昭': 10, '敬': 10, '範': 10, '熙': 10, '에': 10, '베렝가리오': 10, '哲,': 10,
 '清': 10, '恭': 10, '永,': 10, '宣': 10, '勳,': 10, '源': 10, '建': 10, '藤': 10, '촐라': 10,
'織': 10, '弐': 9, '秋': 9, '라는': 9, '넴초프가': 9, '懿': 9, '阿': 9, '彦': 9, '顯': 9, '斗': 9,
'伯': 9, '鎬,': 9, '를': 9, '從': 9, '쥘': 9, '호엔촐레른지크마링겐': 9, '健': 9, '科': 9, ... }
```

- 대부분의 unk 토큰들은 한자로 발생했기에 Data Preprocessing 에서 한자 제거 함수를 만들기로 결정

### 3.1.3. Word Frequency (홍찬우)

- 자주 등장하는 단어들을 유의어로 대체하는 data augmentation 기법을 고려해 단어 빈도수 분석
- Special token 및 따옴표와 같은 기호들을 제외한 단어들을 `단어: 빈도수` 형태로 정리
  ```
  {'한국': 2676, '대한민국': 2348, '대표': 1902, '선수': 1872, 'FC': 1811, '밝혔': 1735, '대통령': 1710,
  '이후': 1639, '광주': 1571, '리그': 1556, '지난': 1496, '미국': 1435, '축구': 1434, '경기': 1417,
  '일본': 1403, '의원': 1348, '말': 1346, '서울': 1325, '지역': 1245, '소속': 1184, '국가': 1175,
  '후보': 1139, '더불': 1130, '코': 1120, '시즌': 1117, '감독': 1093, '국민': 1071, '당시': 1060, ...}
  ```

## 3.2. Preprocessing

### 3.2.1. Chinese-Characters Cleaning (정윤석)

```python
r'([-=+#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》一-鿕㐀-䶵豈-龎\s·])'
```

- 위의 정규식을 이용하여 한자를 감지할 시 한자를 제거하는 함수 생성
- **생성 결과**
  ```python
  Before: 박용오(朴蓉旿, 1937년 4월 29일(음력 3월 19일)(음력 3월 19일) ~ 2009년 11월 4일)는 서울에서 태어난 대한민국의 기업인으로 두산그룹 회장, KBO 총재 등을 역임했다.
  After: 박용오(1937년 4월 29일(음력 3월 19일)(음력 3월 19일) ~ 2009년 11월 4일)는 서울에서 태어난 대한민국의 기업인으로 두산그룹 회장, KBO 총재 등을 역임했다.
  ```

### 3.2.2. Data Augmentation (김세형)

**Easy data augmentation**

- Wei and Zhou (2019)의 4가지 easy data augmentation 방법을 각각의 함수로 구현하여 사용하였으며, augmentation으로 인해 entity의 위치 정보가 변동된 경우 반영하여 수정 (entity 내 단어는 augmentation 대상에서 제외)
  - Synonym replacement (SR): 문장 내 임의의 단어를 유의어로 교체 [그림 3.?]
    ![그림 3.11. SR augmentation 예시(`의회` → `입법부`, `연합` → `동맹`)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/11.png)
    그림 3.11. SR augmentation 예시(`의회` → `입법부`, `연합` → `동맹`)
  - Random deletion (RD): 문장 내 임의의 단어를 삭제 [그림 3.?]
    ![그림 3.12. RD augmentaion 예시(`김두관` 삭제)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/12.png)
    그림 3.12. RD augmentaion 예시(`김두관` 삭제)
  - Random swap (RS): 문장 내 임의의 단어 한 쌍의 위치를 교체 [그림 3.?]
    ![그림 3.?. RS augmentation 예시(`김종석` - `영덕` 교체)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/13.png)
    그림 3.?. RS augmentation 예시(`김종석` - `영덕` 교체)
  - Random insertion (RI): 문장 내 임의의 위치에 임의의 단어를 삽입 [그림 3.?]
    ![그림 3.14. RI augmentation 예시(`통로` 삽입: `팀을 UEFA컵` → `팀을 통로 UEFA컵`)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/14.png)
    그림 3.14. RI augmentation 예시(`통로` 삽입: `팀을 UEFA컵` → `팀을 통로 UEFA컵`)
- SR, RS, RI, RD 순으로 적용 우선순위를 설정(얼마나 문장의 원형이 보존되는지 기준)하고, 데이터 개수에 따라 augmentation 기법의 개수를 차등 적용
  - (0, 100), [100, 200), [200, 450), [450, 700)의 범위에 각각 4, 3, 2, 1개의 기법 적용

**Entity replacement (ER)**

- Entity 정보에 type이 명시되어 있는 점에 착안하여, entity의 단어를 동일한 type을 가진 다른 entity 단어로 교체
  ![그림 3.15. ER augmentation 예시(`subject_entity` 교체)](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/15.png)
  그림 3.15. ER augmentation 예시(`subject_entity` 교체)
  - Threshold(1,000개 or 2,000개)를 설정하여, 개수가 threshold 미만인 데이터를 threshold까지 augmentation. 다만 최대 제한을 augmentation 이전 데이터 개수의 2배로 두어, ER 기법이 각 데이터에 최대 1번 적용되도록 설정

**전체 augmentation 프로세스**

- 최대한 부족한 데이터를 증강시키면서, 과적합을 피할 수 있도록 하는 것이 목표
- 6가지 버전의 데이터(대조군) 생성 및 실험한 결과, 주목할만한 성능 향상은 확인되지 않았음
  - No aug.
    / easy data aug. only
    / ER(thres: 1,000) only
    / easy data aug. + ER 1,000
    / easy data aug. + ER 2,000
    / easy data aug. + ER 2,000 + no_relation 개수 절반 cut

## 3.2.3. Entity Representation (이동호)

- **개요**
  - 본 대회의 Train과 Test dataset에는 Entity의 type이 존재
    ![그림 3.16. dataset의 entity 예시](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/16.png)
    그림 3.16. dataset의 entity 예시
  - 기존 base 코드에서는 type을 활용하지 않음
  - 이를 활용하면 성능 향상을 이룰 수 있을 거라 판단
- **내용**

  - Typed Entity Marker([Zhong and Chen, 2021](https://www.notion.so/KLUE-Wrap-Up-Report-7e063543d6154e02ad26f350bcabe04b?pvs=21))와 Typed Entity Marker (punct)([Zhou and Chen, 2021](https://www.notion.so/KLUE-Wrap-Up-Report-7e063543d6154e02ad26f350bcabe04b?pvs=21)), Sentence Swap 사용
    <aside>
    💡 base

    - [CLS] subj [SEP] obj [SEP] sentence [SEP]

    Sentence Swap

    - base에 [CLS] obj [SEP] subj [SEP] sentence [SEP] 데이터 추가

    Typed Entity Marker

    - sentence에 <S:TYPE> subj </S:TYPE> … <O:TYPE> object_entity </O:TYPE> 형식이 되는 마커를 부착

    Typed Entity Marker (punct)

    - sentence에 @ _ subj-type _ subj @ ... # ∧ obj-type ∧ obj # 형식이 되는 마커를 부착
    </aside>

    - 각각의 marker를 special token으로 넣었을 때와 넣지 않았을 때 비교

- **결과**
  ![그림 3.17 실험 결과](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/17.png)
  그림 3.17 실험 결과
- **결론**
  - klue/bert-base 모델에서는 Typed Entity Marker가 효과적
  - klue/roberta-large의 경우 Typed Entity Marker (punct)가 효과적
  - 이후 모델 학습은 두 가지 방법 중 하나를 적용하여 진행

## 3.3. Model Selection (홍찬우)

<aside>
💡 **모든 모델은 동일 조건에서 실험하고 성능을 비교하기 위해 train, dev dataset 고정 및 hyperparameter 통일**
`epoch=10`, `learning_rate=1e-5`, `batch_size=16,` `warmup_steps=1000`, `optimizer=AdamW`, `scheduler=StepLR`

</aside>

| 모델                                     | 파라미터 수 | F1 / AUPRC (dev) | F1 / AUPRC (public) |
| ---------------------------------------- | ----------- | ---------------- | ------------------- |
| klue/bert-base                           | 125M        | 83.302 / 77.652  |                     |
| klue/roberta-large                       | 355M        | 85.04 / 77.537   |                     |
| xlm-roberta-large                        | 355M        | 84.344 / 78.023  |                     |
| wooy0ng/korquad1-klue-roberta-large      | 355M        | 86.03 / 79.491   | 69,475 / 73.0159    |
| kykim/albert-kor-base                    | 11M         | 79.315 / 65.227  |                     |
| kykim/electra-kor-base                   | 85M         | 77.358 / 49.062  |                     |
| beomi/KcELECTRA-base                     | 85M         | 73.47 / 43.678   |                     |
| snunlp/KR-ELECTRA-discriminator          | 85M         | 79.729 / 63.963  |                     |
| monologg/koelectra-base-v3-discriminator | 85M         | 78.667 / 55.246  |                     |
| skt/kogpt2-base-v2                       | 125M        | 78.174 / 67.998  |                     |
| google/rembert                           | 469M        | 84.84 / 78.663   | 67.5282 / 68.6837   |
| setu4993/LaBSE                           | 470M        | 81.447 / 73.052  |                     |
| timpal01/mdeberta-v3-base-squad2         | 86M         | 79.992 / 61.719  |                     |
| studio-ousia/mluke-large-lite            | 561M        | 85.623 / 79.949  | 69.1844 / 71.59     |
| hfl/cino-large-v2                        | 442M        | 84.624 / 78.901  |                     |

- dev F1 score가 현저히 낮게 나타나는 모델은 따로 public 제출 및 score를 계산하지 않음
- 실험 결과, 모델 파라미터 수가 많을 수록 좋은 성능을 보임
- 그 외 fully connected layer와 연결한 T5, bart model을 시도했으나 실패

**모델 설명**

| 모델 | Description |
| ---- | ----------- |

| RoBERTa
-based | Dynamic masking 기법과 더 많은 데이터를 학습에 활용하여 BERT 모델을 더 강인하게 개선한 모델 |
| ELECTRA
-based | 기존 BERT 계열의 모델들과 달리 대체 토큰 탐지라는 훈련 방식을 통해서 훈련 |
| 기타 | RemBERT (Chung et al., 2020)

- 입력 임베딩과 출력 임베딩 간 가중치 공유를 하지 않음
  MLUKE (Yamada et al.. 2020)
- 단어 시퀀스 이외에 엔티티 시퀀스와 임베딩을 정의
- 단어와 엔티티의 모든 시퀀스 쌍에 따라 별도의 쿼리 파라미터를 두고 셀프 어텐션을 수행
  CINO
- chinese 外 6개 minority languages에 대한 xlm RoBERTa model |

## 3.4. Hyperparameter Tuning (이준선)

### WandB - Sweep을 사용한 Hyperparameter Tuning

![그림 3.18 klue/roberta-large 모델을 sweep을 사용해 학습한 결과](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/18.png)

그림 3.18 klue/roberta-large 모델을 sweep을 사용해 학습한 결과

### Tuning Configuration

- **Learning Rate**
- **Max Epoch**
- **Batch Size**

- **Weight Decay**
- **LR Scheduler**
- **Warmup Steps**

- **Typed-Entity Marker**
- **Augmentation**

## 3.5. Ensemble (이준선)

### 3.5.1. Soft-Voting

```python
dfs = [pd.read_csv(path) for path in model_paths]

probs = []
for row in zip(*[df['probs'].tolist() for df in dfs])
		temp = []
		for col in zip(*[eval(p) for p in row]):
				temp.append(sum(col) / len(col))
		probs.append(temp)

pred_label = [n2l[i.index(max(i))] ofri in probs]
```

- 각 모델의 test data 예측 csv 수집
- `probs`: 각 모델의 class별 예측 확률을 산술 평균
- `pred_label`: 평균낸 확률 중 가장 높은 확률을 가진 class 선택

---

# 4. 수행 결과

## 4.1. Single Models (이준선)

![그림 4.1 단일 모델에 대한 성능표](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/19.png)

그림 4.1 단일 모델에 대한 성능표

## 4.2. Ensemble Models (정윤석)

![그림 4.2 앙상블 조합 성능표](%5BKLUE%5D%20Wrap-Up%20Report%207e063543d6154e02ad26f350bcabe04b/20.png)

그림 4.2 앙상블 조합 성능표

---

# 5. 자체 평가

## 5.1. What’s Good

- Level 2 첫 프로젝트임에도 불구하고 팀에서 계획한 프로세스 그대로 프로젝트를 진행할 수 있었다. 모든 팀원들이 프로세스 전체에 익숙하지는 않았지만 각자 맡은 역할을 충실히 이행하면서 진행이 더뎌진다거나 멈추는 일 없이 진행되었다.
- 대부분의 팀원이 깃헙의 브랜치를 이용하여 협업하는 경험이 적어, 이번 프로젝트에선 merge 같은 것에서 프로젝트 파일이 영향이 가더라도 깃헙 툴에 익숙해지기 위한 경험으로 생각하기로 논의하였기에 모든 팀원들이 부담없이 깃헙을 사용할 수 있었고 이는 귀중한 협업 경험이 되었다.

## 5.2. What’s Bad

- 협업을 할 때 깃헙을 사용하였지만 확실한 기준을 가지고 commit이나 branch 만드는 것이 아닌 개인 별로 작성하였다. 이번 프로젝트 같이 작은 프로젝트에선 상관 없지만 큰 프로젝트라면 문제가 생길만한 부분이었다. 큰 프로젝트를 대비하여 다음 프로젝트부터 깃헙 사용 기준을 마련하여 이에 맞춰 정제된 프로젝트 구조를 만들고 이 구조에 맞춰 프로젝트를 진행할 것이다.
- 많은 모델들은 학습을 성공시켰지만 몇몇 모델들에선 학습이 실패할 때가 있었다. 학습을 실패하는 건 그럴 수 있다 생각하지만 실패한 원인이 왜 생겼는지에 대한 분석이 실패하는 것은 당연한 일이 아니다. 이후 프로젝트에 대비하여 지식을 좀 더 갖춰서 실패 원인 분석을 확실히 하도록 하겠다.

## 5.3. What’s Learned

- 먼저 앙상블 조합을 위한 모델들을 찾는 과정에서 Hugging Face의 다양한 모델들을 학습하는 경험을 할 수 있었다. 그냥 모델 변수의 이름만 바꾸면 학습되는 모델이 있는가 한편 다시 추가 조정이 필요한 모델들도 있어 모델 조작에 익숙해지는 귀중한 경험을 얻을 수 있었다.
- 깃헙을 사용하는 과정에서 문제 없이 파일을 merge하거나 push 할 수 있었지만 때때로 문제가 발생하였다. 이를 해결하는 과정에서 Git 에 대한 개념을 깊이 이해할 수 있었고 이후 같은 문제가 발생 시 이전에 비해 빠르게 해결할 수 있는 능력을 길렀다. 이 같은 능력은 다음 프로젝트 혹은 이후 필드에서 팀원과의 협업을 더욱 수월하게 할 것이다.

---

# Reference

1. [Chung, H. W., Fevry, T., Tsai, H., Johnson, M., & Ruder, S. (2020). Rethinking embedding coupling in pre-trained language models. *arXiv preprint arXiv:2010.12821*.](https://arxiv.org/pdf/2010.12821.pdf)
2. [Wei, J., & Zou, K. (2019). Eda: Easy data augmentation techniques for boosting performance on text classification tasks. *arXiv preprint arXiv:1901.11196*.](https://arxiv.org/pdf/1901.11196)
3. [Yamada, I., Asai, A., Shindo, H., Takeda, H., & Matsumoto, Y. (2020). LUKE: Deep contextualized entity representations with entity-aware self-attention. *arXiv preprint arXiv:2010.01057*.](https://arxiv.org/pdf/2010.01057)
4. [Yang, Z., Xu, Z., Cui, Y., Wang, B., Lin, M., Wu, D., & Chen, Z. (2022). CINO: A Chinese Minority Pre-trained Language Model. *arXiv preprint arXiv:2202.13558*.](https://arxiv.org/pdf/2202.13558)
5. [Zhou, W., & Chen, M. (2021). An improved baseline for sentence-level relation extraction. *arXiv preprint arXiv:2102.01373*](https://arxiv.org/pdf/2102.01373.pdf)
6. [Zhong, Z., & Chen, D. (2020). A frustratingly easy approach for entity and relation extraction. *arXiv preprint arXiv:2010.12812*.](https://aclanthology.org/2021.naacl-main.5.pdf)

---

# 파일 구조

```
level2_klue-nlp-09
|-- README.md
|-- best_model
|-- config
|   `-- config.yaml
|-- preprocessing
|-- eda
|   |-- JYS.ipynb
|   |-- KSH.ipynb
|   |-- LDH.ipynb
|   `-- LJS.ipynb
|-- inference.py
|-- load_data.py
|-- prediction
|-- requirements.txt
|-- requirements_pl.txt
|-- results
|-- train.py
|-- pl_train.py
|-- pl_sweep.py
|-- pl_inference.py
|-- .gitgnore
`-- utils.py
```
