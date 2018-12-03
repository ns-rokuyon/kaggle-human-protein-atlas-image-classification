import numpy as np
from data import *


def get_label_counts():
    df = get_train_df()

    labels = {str(i): 0 for i in range(n_class)}
    for target in df['Target']:
        ts = target.split(' ')
        for t in ts:
            labels[t] += 1

    return labels


def inverted_image_list():
    df = get_train_df()

    inverted = {str(i): set() for i in range(n_class)}
    for i in range(df.shape[0]):
        row = df.iloc[i]
        target = row['Target']
        ts = target.split(' ')
        image_id = row['Id']
        for t in ts:
            inverted[t].add(image_id)

    return inverted