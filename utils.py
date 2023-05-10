import random
import torch
import pandas as pd

# import pytorch_lightning as pl
import numpy as np
import random
import argparse
import yaml
from collections import namedtuple


def seed_everything(seed: int) -> None:
    """
    모든 랜덤 시드를 고정
        Args:
            seed (int): 시드 고정에 사용할 정수값
        Returns:
            None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    # pl.seed_everything(seed, workers=True)


def load_yaml(model_name: str) -> dict:
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default=None)
    # parser.add_argument("--batch_size", default=None)
    # parser.add_argument("--max_epoch", default=None)
    # parser.add_argument("--shuffle", default=None)
    # parser.add_argument("--learning_rate", default=None)
    # parser.add_argument("--train_path", default=None)
    # parser.add_argument("--dev_path", default=None)
    # parser.add_argument("--test_path", default=None)
    # parser.add_argument("--predict_path", default=None)
    # parser.add_argument("--weight_decay", default=None)
    # parser.add_argument("--warm_up_ratio", default=None)
    # parser.add_argument("--loss_func", default=None)
    # parser.add_argument("--run_name", default=None)
    # parser.add_argument("--project_name", default=None)
    # parser.add_argument("--entity", default=None)  # wandb team name
    # args = parser.parse_args()

    with open(f"./config/{model_name}_binary.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # with open(f"./config/klue_roberta_large.yaml") as f:
    #     config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # for key in args.__dict__:
    #     if args.__dict__[key] is not None:
    #         config_dict[key] = args.__dict__[key]

    # learning_rate가 str로 들어오면 float으로 변환
    if isinstance(config_dict["learning_rate"], str):
        config_dict["learning_rate"] = float(config_dict["learning_rate"])

    Config = namedtuple("config", config_dict.keys())
    config = Config(**config_dict)
    print(config)
    return config

def load_yaml_type(model_name: str, type) -> dict:
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", default=None)
    # parser.add_argument("--batch_size", default=None)
    # parser.add_argument("--max_epoch", default=None)
    # parser.add_argument("--shuffle", default=None)
    # parser.add_argument("--learning_rate", default=None)
    # parser.add_argument("--train_path", default=None)
    # parser.add_argument("--dev_path", default=None)
    # parser.add_argument("--test_path", default=None)
    # parser.add_argument("--predict_path", default=None)
    # parser.add_argument("--weight_decay", default=None)
    # parser.add_argument("--warm_up_ratio", default=None)
    # parser.add_argument("--loss_func", default=None)
    # parser.add_argument("--run_name", default=None)
    # parser.add_argument("--project_name", default=None)
    # parser.add_argument("--entity", default=None)  # wandb team name
    # args = parser.parse_args()

    with open(f"./config/{model_name}_{type}.yaml") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    # with open(f"./config/klue_roberta_large.yaml") as f:
    #     config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # for key in args.__dict__:
    #     if args.__dict__[key] is not None:
    #         config_dict[key] = args.__dict__[key]

    # learning_rate가 str로 들어오면 float으로 변환
    if isinstance(config_dict["learning_rate"], str):
        config_dict["learning_rate"] = float(config_dict["learning_rate"])

    Config = namedtuple("config", config_dict.keys())
    config = Config(**config_dict)
    print(config)
    return config

def concat():
  per = pd.read_csv("./prediction/PER.csv")
  org = pd.read_csv("./prediction/ORG.csv")

  concat_df = pd.concat([per, org])

  #concat_df = concat_df.reset_index(drop=True)
  #concat_df.set_index('id', inplace=True)

  no_relation = [1] + [0] * 29
  new_data = pd.DataFrame({'id': [6820], 'pred_label': ['no_relation'], 'probs': [no_relation]})
  concat_df = concat_df.append(new_data)

  sort_df = concat_df.sort_values('id')
  print(sort_df.head())
  # sort_df = sort_df.drop('Unnamed: 0', axis=1)


  #sort_df = sort_df.iloc[:, 1:]

  sort_df.to_csv('./prediction/submission.csv')