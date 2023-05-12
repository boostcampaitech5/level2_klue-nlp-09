import pandas as pd
import pickle
import numpy as np

# num2label 파일 로드
num2label = "./dict_num_to_label.pkl"
with open(num2label, "rb") as f:
    n2l = pickle.load(f)

# 파일 경로
model_paths = [
    "prediction/klue_roberta_large_f1_64.6972.csv",
    "prediction/klue_roberta_large_f1_64.4784.csv",
]

# submission 파일 로드
dfs = [pd.read_csv(path) for path in model_paths]

# probs 합산
probs = []
for row in zip(*[df["probs"].tolist() for df in dfs]):
    temp = []
    for col in zip(*[eval(p) for p in row]):
        temp.append(sum(col) / len(col))
    probs.append(temp)

pred_label = [n2l[i.index(max(i))] for i in probs]

df = pd.DataFrame(columns=["id", "pred_label", "probs"])
df["id"] = range(0, len(pred_label))
df["pred_label"] = pred_label
df["probs"] = probs

df.to_csv(f"prediction/ensemble/{len(model_paths)}.csv", index=False)
