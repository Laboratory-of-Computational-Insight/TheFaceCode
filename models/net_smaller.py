from typing import Optional

import torch.nn as nn
from objects.context import get_context

class Net(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        LAYER_1_CHANNEL = 64
        LAYER_6_CHANNEL = 64

        self.LAYER_1_CHANNEL = LAYER_1_CHANNEL

        self.conv2d_1 = nn.Conv2d(1,LAYER_1_CHANNEL,kernel_size=5,padding=2)
        self.batch_norm1 = nn.BatchNorm2d(LAYER_1_CHANNEL)
        self.maxpool2d_1 = nn.MaxPool2d(kernel_size=2)
        self.dropout_1 = nn.Dropout(p=0.4)

        self.flatten = nn.Flatten()

        self.dense1 = nn.Linear(LAYER_6_CHANNEL*24*24, 128)
        self.batch_norm7 = nn.BatchNorm1d(128)
        self.dropout_4 = nn.Dropout(p=0.6)
        self.out_layer = nn.Linear(128, n_classes)

        self.elu = nn.functional.elu
        self.softmax = lambda x : nn.functional.softmax(x, 1)
        self.history=[]
        self.save_history=False
        self.single_output = False

        self.remove_iden = False

    def toggle_remove_iden(self, remove_iden:Optional[bool]=None):
        if remove_iden is not None:
            self.remove_iden = remove_iden
        else:
            self.remove_iden = not self.remove_iden

    def save_history_flag(self, history:Optional[bool]=None, elements=None):
        if history is not None:
            self.save_history = history
        else:
            self.save_history = not self.save_history
        self.elements = elements

    def forward(self, x, y=None):
        if len(x.shape) ==3:
            x= x.unsqueeze(0)

        history={}
        if self.save_history and (self.elements is None or "original" in self.elements): history["original"] = x.clone().detach().to(get_context().cpu_device)
        x = self.conv2d_1(x)
        if self.save_history and (self.elements is None or "conv2d_1" in self.elements): history["conv2d_1"] = x.clone().detach().to(get_context().cpu_device)
        x = self.elu(x)
        if self.save_history and (self.elements is None or "elu1" in self.elements): history["elu1"] = x.clone().detach().to(get_context().cpu_device)
        x = self.batch_norm1(x)
        if self.save_history and (self.elements is None or "batch_norm1" in self.elements): history["batch_norm1"] = x.clone().detach().to(get_context().cpu_device)
        self.inner_p1 = x

        if self.save_history and (self.elements is None or "elu2" in self.elements): history["elu2"] = x.clone().detach().to(get_context().cpu_device)
        x = self.maxpool2d_1(x)
        if self.save_history and (self.elements is None or "maxpool2d_1" in self.elements): history["maxpool2d_1"] = x.clone().detach().to(get_context().cpu_device)
        x = self.dropout_1(x)
        if self.save_history and (self.elements is None or "dropout_1" in self.elements): history["dropout_1"] = x.clone().detach().to(get_context().cpu_device)

        x= self.flatten(x)
        if self.save_history and (self.elements is None or "flatten" in self.elements): history["flatten"] = x.clone().detach().to(get_context().cpu_device)
        x= self.dense1(x)
        if self.save_history and (self.elements is None or "dense1" in self.elements): history["dense1"] = x.clone().detach().to(get_context().cpu_device)
        x = self.elu(x)
        if self.save_history and (self.elements is None or "elu7" in self.elements): history["elu7"] = x.clone().detach().to(get_context().cpu_device)
        x = self.batch_norm7(x)
        if self.save_history and (self.elements is None or "batch_norm7" in self.elements): history["batch_norm7"] = x.clone().detach().to(get_context().cpu_device)
        x = self.dropout_4(x)
        if self.save_history and (self.elements is None or "dropout_4" in self.elements): history["dropout_4"] = x.clone().detach().to(get_context().cpu_device)
        self.inter_p3 = x


        x = self.out_layer(x)
        if self.save_history and (self.elements is None or "out_layer" in self.elements): history["out_layer"] = x.clone().detach().to(get_context().cpu_device)
        x = self.softmax(x)
        if self.save_history and (self.elements is None or "softmax" in self.elements): history["softmax"] = x.clone().detach().to(get_context().cpu_device)
        if self.save_history: self.history.append(history)

        return x[0] if self.single_output else x
