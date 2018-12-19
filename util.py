import numpy as np
from data import *



def inverted_image_list(df=None):
    if df is None:
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
