import numpy as np
import torch
from sklearn.decomposition import PCA

def pca_lower_dim(data1, n_components=16, **kwargs):
    pca = PCA(n_components=n_components)
    pca1 = pca.fit_transform(data1.to("cpu").reshape(len(data1),-1))
    # print("explained variance", pca.explained_variance_ratio_)
    return pca1

def pca_get_dim(data1, percent=0.85, n_components=None):
    data_reshaped = data1.reshape(len(data1),-1)
    dim0, dim1 = data_reshaped.shape
    sub = int(dim0*0.8)
    data_reshaped = data_reshaped[torch.randperm(sub)]

    if n_components is None:
        pca = PCA(n_components=min(data_reshaped.shape))
    else:
        pca = PCA(n_components=n_components)
    pca.fit(data_reshaped)
    percentage = pca.explained_variance_ratio_
    index, value, minus_one_index, minus_one_value = cumsum_index(percent, percentage)

    return index, value, minus_one_index, minus_one_value


def cumsum_index(percent, percentage):
    cum_sum = np.cumsum(percentage)
    for index, value in enumerate(cum_sum):
        if value >= percent:
            break
    if value < percent:
        print("WARNING: PCA could not find the desired percent, returning max available", percent, value)
    return index, value, index-1, cum_sum[index-1]
