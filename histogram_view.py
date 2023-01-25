import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.data.dataloader import load_data
from options import Options

if __name__ == '__main__':
    opt = Options().parser()

    data_wrap = load_data(opt)

    filename = opt.hist_csv
    if filename is None or not os.path.exists(filename):
        print("File not found: '%s'" % filename)
        exit(0)

    df = pd.read_csv(filename)

    print("Anomaly Statistics")
    print(df[df['labels'] == 1].describe())

    ax = plt.subplot(121)
    sns.histplot(df[df['labels'] == 1], x='scores', hue='labels', color="orange")
    ax.set_title("Anomaly Detection Score")

    ax = plt.subplot(122)
    sns.histplot(df[df['labels'] == 0], x='scores', hue='labels')
    ax.set_title("Normal Detection Score")
    plt.show()

    exit(0)
