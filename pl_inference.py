import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch

import pytorch_lightning as pl

from itertools import chain

import pickle as pickle
import sklearn
from sklearn.metrics import accuracy_score
import numpy as np
import wandb
from utils import seed_everything, load_yaml
from pl_train import RE_Dataset, Dataloader, Model
import torch.nn.functional as F
import re


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


if __name__ == "__main__":
    model_dict = {0: "klue_bert_base", 1: "klue_roberta_large", 2: "snunlp_kr_electra", 3: "xlm_roberta_large", 4: "google_rembert"}
    model_name = model_dict[4]
    # model_name = "pl_test"
    config = load_yaml(model_name)
    # set seed
    seed_everything(config.seed)

    # # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        config.model_name,
        config.per_device_train_batch_size,
        config.shuffle,
        config.train_path,
        config.dev_path,
        config.dev_path,
        config.predict_path,
        config.data_clean,
        config.data_aug,
    )

    # total_steps = warmup_steps = None
    # if args.warm_up_ratio is not None:
    #     total_steps = (15900 // args.batch_size + (15900 % args.batch_size != 0)) * args.max_epoch
    #     warmup_steps = int((15900 // args.batch_size + (15900 % args.batch_size != 0)) * args.warm_up_ratio)

    # 예측할 모델 경로 설정
    # pt 파일인 경우
    # model_path = "/opt/ml/code/best_model/klue_roberta_large/0511_693538.pt"
    # score = "63.9121"
    # ckpt 파일인 경우
    model_path = "./results/google/rembert_0006_val_f1=63.0301.ckpt"
    score = re.search(r"[0-9]{2}\.[0-9]{4}", model_path).group()

    # 저장된 모델로 예측을 진행합니다.
    if model_path.endswith(".pt"):
        model = torch.load(model_path)
    elif model_path.endswith(".ckpt"):
        model = Model.load_from_checkpoint(model_path)

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(accelerator="gpu")  # GPU 사용

    predictions = trainer.predict(model=model, datamodule=dataloader)

    output_pred = []
    output_prob = []

    # 배치로 묶여있으므로 리스트로 풀어준다.
    # predictions shape: (486, 16, 30)
    for i in predictions:
        logits = i
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    pred_answer, output_prob = np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("~/code/prediction/sample_submission.csv")
    output["pred_label"] = pred_answer
    output["probs"] = output_prob
    output.to_csv(f"prediction/{model_name}_f1_{score}.csv", index=False)
