import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    AdamW,
)
from preprocessing.process_manipulator import SequentialCleaning as SC, SequentialAugmentation as SA
from load_data import *

import wandb
from utils import seed_everything, load_yaml


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


def label_to_num(label: list) -> list:
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train(config) -> None:
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "klue/bert-base"
    MODEL_NAME = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data(config.train_path)
    dev_dataset = load_data(config.dev_path)

    #############################
#     cleaning_list = config.data_clean
#     augmentation_list = config.data_aug
    
#     sc = SC(cleaning_list)
#     sa = SA(augmentation_list)
    
#     train_dataset = sc.process(train_dataset)
#     train_dataset = sa.process(train_dataset)
    ################################

    train_label = label_to_num(train_dataset["label"].values)
    dev_label = label_to_num(dev_dataset["label"].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
    print(model.config)
    model.parameters
    model.to(device)

    # init optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma)

    # init wandb logger
    wandb.init(project=config.project_name, name=config.run_name)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir=config.output_dir,  # output directory
        save_total_limit=config.save_total_limit,  # number of total save model.
        save_steps=config.save_steps,  # model saving step.
        num_train_epochs=config.num_train_epochs,  # total number of training epochs
        learning_rate=config.learning_rate,  # learning_rate
        per_device_train_batch_size=config.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.per_device_eval_batch_size,  # batch size for evaluation
        warmup_steps=config.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=config.weight_decay,  # strength of weight decay
        logging_dir=config.logging_dir,  # directory for storing logs
        logging_steps=config.logging_steps,  # log saving step.
        evaluation_strategy=config.evaluation_strategy,  # evaluation strategy to adopt during training
        eval_steps=config.eval_steps,  # evaluation step.
        load_best_model_at_end=config.load_best_model_at_end,
        seed=config.seed,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        optimizers=(optimizer, scheduler),  # define optimizer and scheduler
    )

    # train model
    trainer.train()
    model.save_pretrained(config.save_path)


def main():
    model_dict = {0: "klue_bert_base", 1: "klue_roberta_large", 2: "snunlp_kr_electra"}
    model_name = model_dict[1]
    config = load_yaml(model_name)
    seed_everything(config.seed)
    train(config)


if __name__ == "__main__":
    main()
