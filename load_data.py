import pickle as pickle
import os
import pandas as pd
import torch
import re

from collections import Counter

def num_to_label(num: list) -> list:
    """ 숫자로 된 라벨들을 다시 문자열 형태의 라벨로 변경합니다 """
    str_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in num:
        str_label.append(dict_num_to_label[v])

    return str_label

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class RE_Dataset_Sampler_3(torch.utils.data.IterableDataset):
    """ OverSampling을 활용한 Dataset 구성을 위한 class
    라벨을 no_relation, per, org로 나누어 uniform distribution을 제공합니다
    """

    def __init__(self, pair_dataset, labels):
        """
        Arguments
        ---------
        pair_dataset
            a dictionary of dataset
        labels
            a list of class labels
        """
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.num_labels = len(labels)

        # 숫자로 된 label을 원래의 형태인 str 형태로 변환합니다
        str_labels = num_to_label(labels)

        # parsing을 통해 30 종류의 라벨을 no, per, org 세 종류의 라벨로 변환(비닝)합니다
        new_labels = self.parsing(str_labels)

        # 세 개의 라벨 종류 각각에 대한 개수를 구하고 bin_count에 저장합니다
        bin_count = Counter(new_labels)

        # bin_count를 통해 라벨 데이터 각각에 대해서 weight를 구합니다
        weights = [1.0 / bin_count[label] for label in new_labels]
        self.weights = torch.DoubleTensor(weights)
         

    def __iter__(self):
        
        count = 0
        # torch.multinomial을 통해 weight에 비례하게 index를 뽑고 index_list에 저장합니다
        index_list = [i for i in torch.multinomial(
            self.weights, self.num_labels, replacement=True
        )]
        
        # 아래 print는 multinomial로 인해 실제로 라벨이 바뀌었는지를 확인합니다
        # print("original", str_labels[0])
        # print("changed", num_to_label(torch.tensor(self.labels)[index_list].tolist())[0])
        # num_labels만큼 index_list의 element인 index를 차례대로 넣어 새로운 item을 생성하고 반환합니다
        while count < self.num_labels:
            item = {key: val[index_list[count]].clone().detach() for key, val in self.pair_dataset.items()}
            item["labels"] = torch.tensor(self.labels[index_list[count]])

            yield item
            count += 1

    def __len__(self):
        return len(self.labels)

    def parsing(self, str_labels):
        "30개의 라벨을 no, per, org 3개의 라벨로 변경합니다"
        new_labels = []
        for str_label in str_labels:
            result = re.search(r"(.+):", str_label)
            if result == None:
                new_labels.append("no")
            else:
                new_labels.append(result.group(1))
        return new_labels

class RE_Dataset_Sampler_30(torch.utils.data.IterableDataset):
    """ OverSampling을 활용한 Dataset 구성을 위한 class
    원래의 형태인 30 종류 라벨을 대상으로 uniform distribution을 제공합니다
    """

    def __init__(self, pair_dataset, labels):
        """
        Arguments
        ---------
        pair_dataset
            a dictionary of dataset
        labels
            a list of class labels
        """
        self.pair_dataset = pair_dataset
        self.labels = labels
        self.num_labels = len(labels)

        # 숫자로 된 label을 원래의 형태인 str 형태로 변환합니다
        str_labels = num_to_label(labels)
        new_labels = str_labels

        # 30 개의 라벨 종류 각각에 대한 개수를 구하고 bin_count에 저장합니다
        bin_count = Counter(new_labels)

        # bin_count를 통해 라벨 데이터 각각에 대해서 weight를 구합니다
        weights = [1.0 / bin_count[label] for label in new_labels]
        self.weights = torch.DoubleTensor(weights)
         

    def __iter__(self):
        
        count = 0
        # torch.multinomial을 통해 weight에 비례하게 index를 뽑고 index_list에 저장합니다
        index_list = [i for i in torch.multinomial(
            self.weights, self.num_labels, replacement=True
        )]
        
        # 아래 print는 multinomial로 인해 실제로 라벨이 바뀌었는지를 확인합니다
        # print("original", str_labels[0])
        # print("changed", num_to_label(torch.tensor(self.labels)[index_list].tolist())[0])
        # num_labels만큼 index_list의 element인 index를 차례대로 넣어 새로운 item을 생성하고 반환합니다
        while count < self.num_labels:
            item = {key: val[index_list[count]].clone().detach() for key, val in self.pair_dataset.items()}
            item["labels"] = torch.tensor(self.labels[index_list[count]])

            yield item
            count += 1

    def __len__(self):
        return len(self.labels)

def get_word_new(sentence):
    # ?는 non-greedy matching
    result = re.search(r"'word': '(.+?)'", sentence)
    if result == None:
        result = re.search(r"'word': \"(.+?)\"", sentence)
    result = result.group(1).strip("\"")
    match = re.search(r"\B'\b|\b'\B", result[:-1])
    if match:
        pass
    else:
        result = result.strip("'")
    # result = "'" + result + "'"
    return result

def preprocessing_dataset(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = get_word_new(i)
        j = get_word_new(j)

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {"id": dataset["id"], 
         "sentence": dataset["sentence"], 
         "subject_entity": subject_entity, 
         "object_entity": object_entity, 
         "label": dataset["label"],
        }
    )
    return out_dataset


def preprocessing_dataset_large_class(dataset):
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {"id": dataset["id"], 
         "sentence": dataset["sentence"], 
         "subject_entity": subject_entity, 
         "object_entity": object_entity, 
         "label": dataset["label"][0],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맞게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_data_large_class(dataset_dir):
    """ 
    csv 파일을 경로에 맞게 불러 옵니다. 
    이후, label은 no_relation의 "n", per:~의 "p", org:~의 "o"만 따와 대분류로 설정해줍니다.
    """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset_large_class(pd_dataset)
    
    return dataset


def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity, list(dataset["sentence"]), return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=True,
    )
    
    return tokenized_sentences
