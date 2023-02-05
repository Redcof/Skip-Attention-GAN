import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from lib.data.dataloader import load_data
from options import Options

if __name__ == '__main__':
    opt = Options().parse()

    # data_wrap = load_data(opt)

    filename = opt.hist_csv
    if filename is None or not os.path.exists(filename):
        print("File not found: '%s'" % filename)
        exit(0)

    df = pd.read_csv(filename)

    # for idx, ((batch_x, batch_y), meta) in enumerate(data_wrap.valid):
    #     df.iloc[idx:idx + opt.batchsize, "image_name"] = meta[:, "image"]
    # print("Anomaly Statistics")
    # print(df[df['labels'] == 1].describe())
    # sns.histplot(df, x='scores', hue='labels', kind="kde")

    df['labels'] = df['labels'].apply(lambda x: "Anomaly" if x == 1 else "Normal")

    ax = plt.subplot(111)
    ax.set_title("Anomaly Detection Score")
    sns.histplot(data=df, x="scores", hue="labels", ax=ax, kde=True)

    # ax = plt.subplot(122)
    # ax.set_title("Anomaly Detection Score")
    # sns.kdeplot(
    #     data=df, x="scores", hue="labels",
    #     common_norm=False,
    #     alpha=.5, linewidth=0, ax=ax,
    # )
    plt.show()

    exit(0)
