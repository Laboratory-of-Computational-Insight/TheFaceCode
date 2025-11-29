from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from utils.filesystem.fs import FS
import csv
from enum import Enum, IntEnum

from sklearn.model_selection import train_test_split


class Emotions(IntEnum):
    ANGER=0
    DISGUST=1
    FEAR=2
    HAPPINESS=3
    SADNESS=4
    SURPRISE=5
    NEUTRAL=6

IMAGE_H = 48
IMAGE_W = 48
IMAGE_C = 1


seven_to_three_conversion = {
    3:0,
    4:1,
    6:2,
}

class FER2013DataSet(Dataset):
    def __init__(self, train_val_test, labels=None, split=0.1, random_state = 42, shuffle=True, normalize=True):
        self.data = []
        data_path = FS().get_file("data/all7labels_FER2013_disgustRAF-DB_Affectnet/fer2013.csv")
        df = pd.read_csv(data_path)
        le = LabelEncoder()

        if labels is None:
            labels = list(set(df[df.columns[0]]))
        if normalize:
            df = df[df[df.columns[0]].isin(labels)]
            mapped_labels = le.fit_transform(df[df.columns[0]])
            # le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            df[df.columns[0]] =mapped_labels

        if train_val_test == "test":
            df = df[df[df.columns[2]] == "PrivateTest"]
            for index, row in df.iterrows():
                tensor_data = torch.tensor(np.asarray(row[df.columns[1]].split()).astype(float)).reshape(IMAGE_C,IMAGE_H, IMAGE_W)
                self.data.append(
                    (tensor_data/255,
                     # le_name_mapping[row[df.columns[0]]]
                     row[df.columns[0]]
                     )
                )
        else:
            df = df[df[df.columns[2]] == "Training"]
            X_train, X_valid, y_train, y_valid = train_test_split(df[df.columns[1]], df[df.columns[0]], shuffle=shuffle,test_size=split, random_state=random_state)
            x,y = X_train.iloc(), y_train.iloc()
            if train_val_test == "val":
                x, y = X_valid.iloc(), y_valid.iloc()
            for x_,y_ in zip(x,y):
                tensor_data = torch.tensor(np.asarray(x_.split()).astype(float)).reshape(IMAGE_C, IMAGE_H, IMAGE_W)
                self.data.append(
                    (tensor_data/255,
                     # le_name_mapping[y_]
                     y_
                     )
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FER2013DataLoader:
    def __init__(self):
        pass

    def get_data_loader(self, shuffle=True, batch_size=16, train_val_test: str = "train", labels:Optional[List[int]]=None):
        data = FER2013DataSet(train_val_test, labels)

        return torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )

    def get_train_val_test(self, batch_size, labels:Optional[List[int]]=None):
        train = self.get_data_loader(True, batch_size, "train",labels)
        val = self.get_data_loader(True, batch_size, "val",labels)
        test = self.get_data_loader(False, batch_size, "test",labels)
        return train, val, test