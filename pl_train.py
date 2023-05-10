import pandas as pd
from tqdm.auto import tqdm

import transformers
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from itertools import chain

import pickle as pickle
import sklearn
from sklearn.metrics import accuracy_score
import numpy as np
import wandb
from utils import seed_everything, load_yaml
import re


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=256)
        # self.target_columns = ["label"]
        # self.delete_columns = ["id"]
        # self.text_columns = ["sentence_1", "sentence_2"]

    def tokenizing(self, dataset):
        """tokenizer에 따라 sentence를 tokenizing 합니다."""
        concat_entity = []
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = e01 + "[SEP]" + e02
            concat_entity.append(temp)
        tokenized_sentences = self.tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
        return tokenized_sentences

    def preprocessing(self, dataset):
        """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""

        def get_word_new(sentence):
            # ?는 non-greedy matching
            result = re.search(r"'word': '(.+?)'", sentence)
            if result == None:
                result = re.search(r"'word': \"(.+?)\"", sentence)
            result = result.group(1).strip('"')
            match = re.search(r"\B'\b|\b'\B", result[:-1])
            if match:
                pass
            else:
                result = result.strip("'")
            # result = "'" + result + "'"
            return result

        subject_entity = []
        object_entity = []
        for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
            i = get_word_new(i)
            j = get_word_new(j)

            subject_entity.append(i)
            object_entity.append(j)
        out_dataset = pd.DataFrame(
            {
                "id": dataset["id"],
                "sentence": dataset["sentence"],
                "subject_entity": subject_entity,
                "object_entity": object_entity,
                "label": dataset["label"],
            }
        )
        return out_dataset

    def setup(self, stage=None):
        def label_to_num(label: list) -> list:
            num_label = []
            with open("dict_label_to_num.pkl", "rb") as f:
                dict_label_to_num = pickle.load(f)
            for v in label:
                num_label.append(dict_label_to_num[v])

            return num_label

        if stage == "fit" or stage is None:
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_dataset = pd.read_csv(self.train_path)
            dev_dataset = pd.read_csv(self.dev_path)

            train_dataset = self.preprocessing(train_dataset)
            dev_dataset = self.preprocessing(dev_dataset)

            train_label = label_to_num(train_dataset["label"].values)
            dev_label = label_to_num(dev_dataset["label"].values)

            tokenized_train = self.tokenizing(train_dataset)
            tokenized_dev = self.tokenizing(dev_dataset)

            # 학습데이터 준비
            # self.train_inputs, self.train_targets = self.preprocessing(train_data)
            self.train_dataset = RE_Dataset(tokenized_train, train_label)

            # 검증데이터 준비
            self.dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        else:
            # 평가데이터 준비
            test_dataset = pd.read_csv(self.test_path)
            test_dataset = self.preprocessing(test_dataset)
            test_label = label_to_num(test_dataset["label"].values)
            tokenized_test = self.tokenizing(test_dataset)
            self.test_dataset = RE_Dataset(tokenized_test, test_label)

            predict_dataset = pd.read_csv(self.predict_path)
            predict_dataset = self.preprocessing(predict_dataset)
            predict_label = predict_dataset["label"].values
            tokenized_predict = self.tokenizing(predict_dataset)
            self.predict_dataset = RE_Dataset(tokenized_predict, predict_label)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)


def klue_re_micro_f1(preds, labels) -> float:
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)

    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels) -> float:
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred) -> dict:
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


class Model(pl.LightningModule):
    def __init__(self, model_name, lr, weight_decay, warmup_steps):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        self.model_config = transformers.AutoConfig.from_pretrained(self.model_name)
        self.model_config.num_labels = 30
        # 사용할 모델을 호출
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        # Loss 계산을 위해 사용될 손실함수를 호출
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        input_ids = x["input_ids"]
        token_type_ids = x["token_type_ids"]
        attention_mask = x["attention_mask"]
        logits = self.plm(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)["logits"]

        return logits

    def training_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]
        logits = self(x)
        loss = self.loss_func(logits.float(), y)
        self.log("train_loss", loss)

        f1 = klue_re_micro_f1(logits.argmax(-1).cpu(), y.cpu())
        self.log("train_f1", f1)
        # auprc = klue_re_auprc(logits.cpu(), y.cpu())
        # self.log("train_auprc", auprc)

        return loss

    def validation_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]

        logits = self(x)

        loss = self.loss_func(logits.float(), y)
        self.log("val_loss", loss)

        f1 = klue_re_micro_f1(logits.argmax(-1).cpu(), y.cpu())
        self.log("val_f1", f1)
        # auprc = klue_re_auprc(logits.cpu(), y.cpu())
        # self.log("val_auprc", auprc)

        return loss

    def test_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]
        logits = self(x)

    def predict_step(self, batch, batch_idx):
        x = {"input_ids": batch["input_ids"], "token_type_ids": batch["token_type_ids"], "attention_mask": batch["attention_mask"]}
        y = batch["labels"]
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # warmup stage 있는 경우
        if self.warmup_steps is not None:
            scheduler = transformers.get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=self.warmup_steps)
            return (
                [optimizer],
                [
                    {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "reduce_on_plateau": False,
                        "monitor": "val_loss",
                    }
                ],
            )
        # warmup stage 없는 경우
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
            return [optimizer], [scheduler]


if __name__ == "__main__":
    # model_dict = {0: "klue_bert_base", 1: "klue_roberta_large", 2: "snunlp_kr_electra"}
    # model_name = model_dict[0]
    model_name = "pl_test"
    config = load_yaml(model_name)
    # set seed
    seed_everything(config.seed)

    # actual model train
    # wandb logger
    wandb_logger = WandbLogger(
        project=config.project_name,
        name=config.run_name,
    )

    # # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        config.model_name, config.per_device_train_batch_size, True, config.train_path, config.dev_path, config.dev_path, config.predict_path
    )

    # total_steps = warmup_steps = None
    # if args.warm_up_ratio is not None:
    #     total_steps = (15900 // args.batch_size + (15900 % args.batch_size != 0)) * args.max_epoch
    #     warmup_steps = int((15900 // args.batch_size + (15900 % args.batch_size != 0)) * args.warm_up_ratio)

    model = Model(
        config.model_name,
        config.learning_rate,
        config.weight_decay,
        config.warmup_steps,
    )

    # model = torch.load('model.pt')

    # gpu가 없으면 accelerator='cpu', 있으면 accelerator='gpu'
    trainer = pl.Trainer(
        precision="16-mixed",  # 16-bit mixed precision
        accelerator="gpu",  # GPU 사용
        # dataloader를 매 epoch마다 reload해서 resampling
        # reload_dataloaders_every_n_epochs=1,
        max_epochs=config.num_train_epochs,  # 최대 epoch 수
        logger=wandb_logger,  # wandb logger 사용
        log_every_n_steps=1,  # 1 step마다 로그 기록
        val_check_interval=0.25,  # 0.25 epoch마다 validation
        check_val_every_n_epoch=1,  # val_check_interval의 기준이 되는 epoch 수
        callbacks=[
            # learning rate를 매 step마다 기록
            LearningRateMonitor(logging_interval="step"),
            #     EarlyStopping("val_pearson", patience=8, mode="max", check_finite=False),  # validation pearson이 8번 이상 개선되지 않으면 학습을 종료
            #     CustomModelCheckpoint("./save/", "snunlp_MSE_002_{val_pearson:.4f}", monitor="val_pearson", save_top_k=1, mode="max"),
        ],
    )

    # use Tuner to get optimized batch size
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model=model, datamodule=dataloader, mode="binsearch")

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # # 학습이 완료된 모델을 저장합니다.
    torch.save(model, "model.pt")
    # model.save_pretrained(config.save_path)

# TODO: auprc, accuracy 적용
# TODO: 기타 기능 적용
# TODO: 얼리스탑핑 적용
