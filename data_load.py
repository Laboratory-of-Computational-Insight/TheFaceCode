import os

import numpy as np
import torch

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from scipy.io import loadmat
from sklearn.metrics import silhouette_score

from objects.label_types import Attribute
from utils.filesystem.fs import FS
from utils.init_system import init # must import

DISFA = None
data = None
labels = None
def disfa_train():

    # Cannot share Disfa data due to copyright limitations.
    # Please refer to the original DISFA dataset for labels and data usage.
    # https://mohammadmahoor.com/pages/databases/disfa/
    # this function gets the data as a np.array of dim (num_samples, height, width) and labels as a list of tuples
    # as follow: (0-index, 1-identity, 2-emotion, 3-valence, 4-arousal)
    # the images are originally larger than 48x48 but have been resized for faster processing, and rgb have been removed
    # to grayscale

    faces_disfa = []
    labels = []

    faces_disfa = torch.tensor(faces_disfa)
    faces_disfa = faces_disfa.unsqueeze(dim=1)
    faces_disfa = faces_disfa
    labels = [int(ide) for ind, ide, emo, val, aro in labels]
    grouped = {}

    for face, label in zip(faces_disfa, labels):
        grouped[label] = grouped.get(label, [])
        grouped[label].append(face)

    return grouped


def neg_labels(labels):
    half_n = len(labels)//2
    other_half_n = len(labels) - half_n
    if 0 > labels[0, 1]:
        labels[:half_n,1] = -torch.zeros(half_n)
        labels[half_n:,1] = -torch.ones(other_half_n)


def disfa():

    # Cannot share Disfa data due to copyright limitations.
    # Please refer to the original DISFA dataset for labels and data usage.
    # https://mohammadmahoor.com/pages/databases/disfa/
    # this function gets the data as a np.array of dim (num_samples, height, width) and labels as a list of tuples
    # as follow: (0-index, 1-identity, 2-emotion, 3-valence, 4-arousal)
    # the images are originally larger than 48x48 but have been resized for faster processing, and rgb have been removed
    # to grayscale

    faces_disfa = []
    labels = []


    neg_labels(labels)
    labels = {
        int(ind): (int(ide), int(emo if emo < 6 else 6), val, aro) for ind, ide, emo, val, aro in labels
    }
    faces_disfa = np.vstack((faces1, faces2))

    faces_disfa = torch.tensor(faces_disfa)
    faces_disfa = faces_disfa.unsqueeze(dim=1)
    faces_disfa = faces_disfa

    indexes = np.arange(len(faces_disfa))
    np.random.shuffle(indexes)
    indexes = indexes[:int(len(faces_disfa)*0.8)]
    indexes = sorted(indexes)
    faces_disfa = faces_disfa[indexes]
    new_labels = {}
    for i, index in enumerate(indexes):
        new_labels[i] = labels[index]
    labels = new_labels


    from sklearn.decomposition import PCA
    shape = faces_disfa.shape
    faces_disfa = faces_disfa.reshape(faces_disfa.shape[0], -1)
    pca = PCA(16)
    pca.fit(faces_disfa)
    lower = (faces_disfa - pca.mean_) @ pca.components_.T
    reconstruct = lower @ pca.components_ + pca.mean_
    new_faces = (faces_disfa - reconstruct)
    sil = silhouette_score(new_faces, [labels[l][Attribute.IDENTITY] for l in labels])
    print("sil score of images without the first 16 PC values:", sil)
    print("first 4 pc values explained variance", sum(pca.explained_variance_ratio_[:16]))

    pca_normal = PCA(4)
    pca_normal.fit(faces_disfa)
    comp = pca_normal.components_
    mean = pca_normal.mean_
    np.savez('pca_components_and_mean.npz', comp=comp, mean=mean)


    new_faces = new_faces.reshape(*shape)
    faces_disfa = faces_disfa.reshape(*shape).to(torch.float32)

    agg_new_faces = []
    faces = []

    avg_img = faces_disfa.mean(0).detach().cpu().squeeze().numpy()
    os.makedirs("./plots", exist_ok=True)
    plt.imshow(avg_img, cmap='gray')
    plt.axis('off')
    plt.savefig("./plots/avg_face.png", bbox_inches='tight', pad_inches=0)
    plt.close()


    from sklearn.decomposition import PCA
    shape = faces_disfa.shape
    for f in faces_disfa:
        faces.append(f)

    faces_disfa = faces_disfa.reshape(faces_disfa.shape[0], -1)
    pca = PCA(16)
    pca.fit(faces_disfa)
    mean = torch.from_numpy(pca.mean_).to(torch.float32)
    comp = torch.from_numpy(pca.components_).to(torch.float32)
    lower = (faces_disfa - mean) @ comp.T
    reconstruct = lower @ comp + mean
    new_faces = (faces_disfa - reconstruct)
    new_faces = new_faces.reshape(*shape)

    for f in faces:
        shape_f = f.shape
        f = f.flatten()
        lower_f = (f - mean) @ comp.T
        reconstruct_f = lower_f @ comp + mean
        agg_new_faces.append((f - reconstruct_f).reshape(*shape_f))
    agg_new_faces = torch.stack(agg_new_faces)
    print(agg_new_faces.shape)

    faces_disfa = faces_disfa.reshape(*shape).to(torch.float32) # average-face removing reconstruction
    # 1) Mean of new_faces
    img1 = new_faces.mean(0).detach().cpu().squeeze().numpy()
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.savefig("./plots/avg_new_faces.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # 2) Mean of agg_new_faces
    img2 = agg_new_faces.mean(0).detach().cpu().squeeze().numpy() # average of individual faces without reconstruction
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.savefig("./plots/avg_agg_new_faces.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # 3) Mean of reconstructed faces
    img3 = reconstruct.reshape(*shape).mean(0).detach().cpu().squeeze().numpy()
    plt.imshow(img3, cmap='gray')
    plt.axis('off')
    plt.savefig("./plots/avg_reconstruct.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    # Image processing using PyTorch
    image = new_faces.mean(0).cpu().numpy()
    mask = reconstruct.view(*shape).mean(0).cpu().numpy()
    mask = (mask - mask.min(axis=(1, 2), keepdims=True)) / (
                mask.max(axis=(1, 2), keepdims=True) - mask.min(axis=(1, 2), keepdims=True))
    print(mask.shape)
    # Apply color map using matplotlib
    cmap = plt.cm.jet
    norm = Normalize(vmin=mask.min(), vmax=mask.max())
    mask_colored = cmap(norm(mask))[..., :3]  # Apply colormap and remove alpha channel
    mask = torch.tensor(mask_colored).squeeze(0)
    mask_colored = mask_colored.squeeze(0)
    mask_colored = torch.tensor(mask_colored).permute(2, 0, 1)  # Convert to PyTorch tensor and permute dimensions

    # Overlay the heatmap on the image
    print(image.shape)
    image = torch.tensor(image).repeat(3, 1, 1)
    overlay = 0.6 * image + 0.4 * mask_colored

    # Save both overlay and mask data for plot_compact.py
    np.save("G:\FACE_VEC\custom\plotting_for_paper\overlay.npy", overlay.numpy())
    np.save("G:\FACE_VEC\custom\plotting_for_paper\mask.npy", mask.numpy())
    print(f"Overlay saved to: G:\FACE_VEC\custom\plotting_for_paper\overlay.npy")
    print(f"Mask saved to: G:\FACE_VEC\custom\plotting_for_paper\mask.npy")

    fig, ax = plt.subplots()
    cax = ax.imshow(overlay.permute(1, 2, 0).numpy())
    ax.axis('off')

    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical')
    cbar.set_label('Color Intensity')

    fig.savefig("./plots/overlay.png", bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return faces_disfa, labels


def fer2013():
    # Cannot share FER2013 data due to copyright limitations.
    # Please refer to the original FER2013 dataset for labels and data usage.

    file = ":("
    data = loadmat(file)
    labels = data["labels"]  # 0-index, 1-identity, 2-emotion_classified, 3-emotion_orig, 4-valence, 5-arousal
    neg_labels(labels)
    labels = {
        int(ind): (ide, int(emo_class), int(emo_orig), val, aro) for ind, ide, emo_class, emo_orig, val, aro in labels
    }

    data = torch.tensor(data["data"])

    return data/255, labels


if __name__ == "__main__":
    disfa()