import os

import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


import numpy as np

DARK24 = px.colors.qualitative.Dark24
GREy = px.colors.sequential.Greys
import plotly.io as pio
pio.renderers.default = "browser"

def remove_outlayers(data, colors=None, symbols=None, groups=None):
    indexes_to_remove = set()
    for i in range(data.shape[1]):
        max_ = data[:,i].argmax()
        min_ = data[:, i].argmin()
        indexes_to_remove.add(max_)
        indexes_to_remove.add(min_)

    indexes = [ i for i in range(len(data)) if i not in indexes_to_remove]
    new_data = data[indexes]
    new_colors = None if colors is None else [colors[i] for i in indexes]
    new_symbols = None if symbols is None else [symbols[i] for i in indexes]
    new_groups = None
    if groups is not None:
        new_groups = []
        for name, group in groups:
            new_groups.append((name, [group[i] for i in indexes]))

    return new_data, new_colors, new_symbols, new_groups

def plot_embedding(data,title="", colors=None):
    columns = ["x","y","color"]
    rows = []

    data, colors, _, _ = remove_outlayers(data, colors, None, None)
    data, colors, _, _ = remove_outlayers(data, colors, None, None)


    for i, d in enumerate(data):
        row = [d[0], d[1]]
        row.append(colors[i] if colors else "0")
        rows.append(row)
    df = pd.DataFrame(rows, columns=columns)
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color",
        color_discrete_sequence=DARK24,
        color_continuous_scale="Greys",
        category_orders={"color": [str(i) for i in range(int(df["color"].astype(np.double).min()), 1+int(df["color"].astype(np.double).max()))]},
        title=title
    )
    os.makedirs("./plots", exist_ok=True)
    safe_title = title.replace(" ", "_").replace("/", "_")
    path = f"./plots/{safe_title}.html"
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)


