import numpy as np
from pathlib import Path
import argparse

import pandas as pd
import torch
from scipy.io import loadmat, savemat
from torchvision import transforms

from config import configuration
from emonet.models import EmoNet
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from utils.filesystem.fs import FS
from utils.init_system import init




torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 16
device = 'cuda:0'
image_size = 256
subset = 'test'
metrics_valence_arousal = {'CCC':CCC, 'PCC':PCC, 'RMSE':RMSE, 'SAGR':SAGR}
metrics_expression = {'ACC':ACC}


# Loading the model
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to("cuda")
net.load_state_dict(state_dict, strict=False)
net.eval()

def fer2013():# 1x48x48 not centered
    file = FS().get_file("data/all7labels_FER2013_disgustRAF-DB_Affectnet/fer2013.csv") # Todo remove path
    df = pd.read_csv(file)
    label = df["emotion"]
    data = df["pixels"]

    outputs = []
    trans = transforms.Resize(size=(image_size, image_size))
    imgs = []
    for i in range(len(data)):
        img = data[i]
        img = torch.tensor(np.asarray(img.split()).astype(float))
        img = torch.tensor(img).reshape(48, 48)
        img = img.float().unsqueeze(0)
        imgs.append(img.clone())
        img = img/255
        img = trans(img)
        img = img.repeat(1, 3, 1, 1).to("cuda")

        out = net(img, mode=1)
        net.history.clear()
        emotion = out["expression"].argmax()
        outputs.append(
            [i, -1, emotion.item(), label[i], out["valence"][0].item(), out["arousal"][0].item()]
        )
    data = dict(labels=outputs, data=[img.numpy() for img in imgs])
    data_to_save = savemat("fer2013_labeled.mat", data) # please save to wasabi
fer2013()


def disfa(): #1x48x48
    # assumes you already downloaded the files using FS().get_file. see main_plot.py
    file = FS().get_file("data/DISFA_FEMALES/faces_f12ID_48_48_10skip.mat") # Todo remove path
    faces1=loadmat(file)
    faces1=faces1['data']
    file = FS().get_file('data/DISFA_MALES/Males/faces_disfa_males_48_48.mat') # Todo remove path
    faces2=loadmat(file)
    faces2=faces2['data']
    faces_disfa = np.vstack((faces1,faces2))
    faces_disfa = torch.tensor(faces_disfa)
    trans = transforms.Resize(size=(image_size,image_size))

    faces_disfa=trans(faces_disfa)

    faces_disfa = faces_disfa.unsqueeze(dim=1)

    faces_indexes=[0,485, 970, 1454, 1935, 2419, 2904, 3388, 3873, 4357, 4842, 5326, 5811, 6296, 6780, 7265, 7751, 8236, 8721, 9205, 9690, 10174, 10659, 11143, 11628]
    def color_by_index(index, faces_indexes):
        i=0
        for i, face in enumerate(faces_indexes[1:]):
            if index < face:
                break
        return i

    outputs =[]
    for i in range(len(faces_disfa)):
        img = faces_disfa[i:i+1].to("cuda")
        out = net(img)
        disfa_conversion={ #emo_net: our_model
        0:6, #neutral
        1:3, #Happy
        2:4,#Sad
        3:5,#Surprise
        4:2,#Fear
        5: 1,#Disgust
        6:0,#Anger
        7:7,#Contempt
        }
        emotion =out["expression"].argmax()
        outputs.append(
            [i, color_by_index(i, faces_indexes), disfa_conversion[emotion.item()], out["valence"][0].item(), out["arousal"][0].item()]
        )
    faces_disfa = np.vstack((faces1,faces2))
    data = dict(data=faces_disfa,labels=outputs)
    data_to_save = savemat(data, "disfa_labeled.mat") # please save to wasabi

disfa()
