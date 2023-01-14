import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('hist_csv', default=None, help='Path to exp_histogram.csv')
    opt = parser.parse_args()

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
