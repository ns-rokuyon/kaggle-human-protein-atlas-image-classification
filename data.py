import sys
import cv2
import math
import tqdm
import torch
import random
import h5py
import requests
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from multiprocessing.pool import Pool
from sklearn.model_selection import train_test_split, KFold
from skmultilearn.model_selection import IterativeStratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from optim import CosineLRWithRestarts
try:
    from workspace import *
    on_colab = False
    print('workspace: local')
except ImportError:
    from workspace_colab import *
    on_colab = True
    print('workspace: colab')


progress_bar = tqdm.tqdm if on_colab else tqdm.tqdm_notebook


# Green filter for the target protein structure of interest
# Blue landmark filter for the nucleus
# Red landmark filter for microtubules
# Yellow landmark filter for the endoplasmatic reticulum
colors = ['red','green','blue','yellow']
name_label_dict = {
    0:  'Nucleoplasm',
    1:  'Nuclear membrane',
    2:  'Nucleoli',   
    3:  'Nucleoli fibrillar center',
    4:  'Nuclear speckles',
    5:  'Nuclear bodies',
    6:  'Endoplasmic reticulum',   
    7:  'Golgi apparatus',
    8:  'Peroxisomes',
    9:  'Endosomes',
    10:  'Lysosomes',
    11:  'Intermediate filaments',
    12:  'Actin filaments',
    13:  'Focal adhesion sites',   
    14:  'Microtubules',
    15:  'Microtubule ends',  
    16:  'Cytokinetic bridge',   
    17:  'Mitotic spindle',
    18:  'Microtubule organizing center',  
    19:  'Centrosome',
    20:  'Lipid droplets',
    21:  'Plasma membrane',   
    22:  'Cell junctions', 
    23:  'Mitochondria',
    24:  'Aggresome',
    25:  'Cytosol',
    26:  'Cytoplasmic bodies',   
    27:  'Rods & rings' 
}

n_class = len(name_label_dict)

ex_image_base_url = 'http://v18.proteinatlas.org/images/'
weak_classes = {'8', '9', '10', '15', '17', '20', '24', '26', '27'}


class Stats:
    mean = np.array([0.08069, 0.05258, 0.05487, 0.08282])
    std = np.array([0.13704, 0.10145, 0.15313, 0.13814])


def gen_images_h5_file(df):
    if images_h5_file.exists():
        print(f'Found: {images_h5_file}')
        return

    with h5py.File(str(images_h5_file), 'a') as fp:
        for i in progress_bar(range(df.shape[0])):
            image_id = df.iloc[i]['Id']
            im = load_4ch_image_train(image_id)

            im = np.array(im)

            key = 'train/{}'.format(str(image_id))
            fp.create_dataset(key, data=im)


def gen_ex_images_h5_file(ex_ids):
    """
    >>> ex_ids = get_ex_ids_to_save()
    >>> gen_ex_images_h5_file(ex_ids)
    """
    if ex_images_h5_file.exists():
        print(f'Found: {ex_images_h5_file}')
        return

    with h5py.File(str(ex_images_h5_file), 'a') as fp:
        for image_id in progress_bar(ex_ids):
            im = load_4ch_image_ex(image_id)

            im = np.array(im)

            key = 'ex/{}'.format(str(image_id))
            fp.create_dataset(key, data=im)


def gen_test_images_h5_file(df):
    if test_images_h5_file.exists():
        print(f'Found: {test_images_h5_file}')
        return

    with h5py.File(str(test_images_h5_file), 'a') as fp:
        for i in progress_bar(range(df.shape[0])):
            image_id = df.iloc[i]['Id']
            im = load_4ch_image_test(image_id)

            im = np.array(im)

            key = 'test/{}'.format(str(image_id))
            fp.create_dataset(key, data=im)


def open_images_h5_file():
    return h5py.File(str(images_h5_file), 'r')


def open_ex_images_h5_file():
    return h5py.File(str(ex_images_h5_file), 'r')


def open_test_images_h5_file():
    return h5py.File(str(test_images_h5_file), 'r')


def save_model(model, keyname, optimizer=None, scheduler=None):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))

    if optimizer:
        optim_filename = f'{keyname}_dict.optim'
        torch.save(optimizer.state_dict(), str(model_dir / optim_filename))

    if scheduler:
        scheduler_filename = f'{keyname}_dict.scheduler'
        torch.save(scheduler.state_dict(), str(model_dir / scheduler_filename))


def save_bestmodel(model, keyname):
    dict_filename = f'{keyname}_dict.bestmodel'
    torch.save(model.state_dict(), str(model_dir / dict_filename))


def save_checkpoint(model, keyname, optimizer=None, scheduler=None):
    dict_filename = f'{keyname}_dict.model'
    torch.save(model.state_dict(), str(model_dir / dict_filename))

    if optimizer:
        optim_filename = f'{keyname}_dict.optim'
        torch.save(optimizer.state_dict(), str(model_dir / optim_filename))

    if scheduler:
        scheduler_filename = f'{keyname}_dict.scheduler'
        torch.save(scheduler.state_dict(), str(model_dir / scheduler_filename))
    

def get_train_df():
    df = pd.read_csv(train_csv)
    return df


def get_test_df():
    df = pd.read_csv(test_csv)
    return df


def get_train_val_df_fold(k):
    train_listfile = str(kfold_cv5_list_dir / f'train_cv{k}.csv')
    val_listfile = str(kfold_cv5_list_dir / f'val_cv{k}.csv')
    train_df = pd.read_csv(str(train_listfile))
    val_df = pd.read_csv(str(val_listfile))
    return train_df, val_df


def get_multilabel_stratified_train_val_df_fold(k):
    train_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'train_cv{k}.csv')
    val_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'val_cv{k}.csv')
    train_df = pd.read_csv(str(train_listfile))
    val_df = pd.read_csv(str(val_listfile))
    return train_df, val_df


def get_multilabel_stratified_train_val_df_fold_v2(k):
    train_listfile = str(multilabel_stratified_kfold_cv5_list_v2_dir / f'train_cv{k}.csv')
    val_listfile = str(multilabel_stratified_kfold_cv5_list_v2_dir / f'val_cv{k}.csv')
    train_df = pd.read_csv(str(train_listfile))
    val_df = pd.read_csv(str(val_listfile))
    return train_df, val_df


def create_binary_classification(train_df, val_df, positive_class):
    train_df = train_df.copy()
    val_df = val_df.copy()

    positive_class = str(positive_class)

    def _func(target):
        ts = target.split(' ')
        return '1' if positive_class in ts else '0'

    train_df['Target'] = train_df['Target'].apply(_func)
    val_df['Target'] = val_df['Target'].apply(_func)


def _kfold_dfs(k=5):
    df = get_train_df()
    kf = KFold(n_splits=k, shuffle=True, random_state=1234)
    for train_index, val_index in kf.split(df.index.values):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_kfold_cv5_list():
    for i, (train_df, val_df) in enumerate(_kfold_dfs(k=5)):
        train_listfile = str(kfold_cv5_list_dir / f'train_cv{i}.csv')
        val_listfile = str(kfold_cv5_list_dir / f'val_cv{i}.csv')
        train_df.to_csv(train_listfile, index=False)
        val_df.to_csv(val_listfile, index=False)
        print(f'Generated: {train_listfile}')
        print(f'Generated: {val_listfile}')


def _multilabel_stratified_kfold_dfs():
    df = get_train_df()
    label_mat = multilabel_binary_representation(df, sparse=True)

    kf = IterativeStratification(random_state=1234)  # k=3
    for train_index, val_index in kf.split(df.index.values, label_mat):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_multilabel_stratified_kfold_cv3_list():
    for i, (train_df, val_df) in enumerate(_multilabel_stratified_kfold_dfs()):
        train_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'train_cv{i}.csv')
        val_listfile = str(multilabel_stratified_kfold_cv3_list_dir / f'val_cv{i}.csv')
        train_df.to_csv(train_listfile, index=False)
        val_df.to_csv(val_listfile, index=False)
        print(f'Generated: {train_listfile}')
        print(f'Generated: {val_listfile}')


def _multilabel_stratified_kfold_dfs_v2():
    df = get_train_df()
    label_mat = multilabel_binary_representation(df, sparse=False)

    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # k=3
    for train_index, val_index in kf.split(df.index.values, label_mat):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_multilabel_stratified_kfold_cv5_list_v2():
    for i, (train_df, val_df) in enumerate(_multilabel_stratified_kfold_dfs_v2()):
        train_listfile = str(multilabel_stratified_kfold_cv5_list_v2_dir / f'train_cv{i}.csv')
        val_listfile = str(multilabel_stratified_kfold_cv5_list_v2_dir / f'val_cv{i}.csv')
        train_df.to_csv(train_listfile, index=False)
        val_df.to_csv(val_listfile, index=False)
        print(f'Generated: {train_listfile}')
        print(f'Generated: {val_listfile}')


def load_4ch_image(base_path, id):
    """Get 4ch PIL.Image
    """
    bands = [Image.open(str(base_path / f'{id}_{color}.png'))
             for color in colors]
    image = Image.merge('RGBA', bands=bands)
    return image


def load_4ch_image_train(id):
    return load_4ch_image(train_image_dir, id)


def load_4ch_image_test(id):
    return load_4ch_image(test_image_dir, id)


def load_4ch_image_ex(id):
    return load_4ch_image(ex_image_dir, id)


def open_rgby(base_path, id):
    """function that reads RGBY image

    Return
    ------
    nd.array shape(512, 512, 4)
    """
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(str(base_path / f'{id}_{color}.png'), flags).astype(np.float32) / 255
           for color in colors]
    return np.stack(img, axis=-1)


def open_rgby_train(id):
    return open_rgby(train_image_dir, id)


def open_rgby_test(id):
    return open_rgby(test_image_dir, id)


def get_class_weights(df, max_value=100):
    pos = [0 for c in range(n_class)]
    neg = [0 for c in range(n_class)]
    for i in progress_bar(range(df.shape[0])):
        labels = [int(label) for label in df.iloc[i]['Target'].split(' ')]
        for c in range(n_class):
            if c in labels:
                pos[c] += 1
            else:
                neg[c] += 1
    weights = [n / p for p, n in zip(pos, neg)]
    if max_value is not None:
        weights = [w if w < max_value else max_value for w in weights]
    return weights


def write_log(message, keyname):
    print(message)
    with open(str(log_dir / f'{keyname}.txt'), 'a') as fp:
        print(message, file=fp)


def multilabel_binary_representation(df, sparse=True):
    mat = []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        target = row['Target']
        ts = set(map(int, target.split(' ')))

        label_array = [1 if c in ts else 0 for c in range(n_class)]
        mat.append(label_array)

    mat = np.array(mat)
    if sparse:
        return sp.lil_matrix(mat)

    return mat


def create_sampling_log_weights(df, mu=0.5):
    label_set_count = defaultdict(int)
    for i in range(df.shape[0]):
        target = str(df.iloc[i]['Target'])
        label_set_count[target] += 1

    label_set_weights = {target: math.log(mu * df.shape[0] / count)
                         for target, count in label_set_count.items()}
    weights = [label_set_weights[str(df.iloc[i]['Target'])]
               for i in range(df.shape[0])]
    return weights, label_set_count


def create_weighted_random_sampler(df):
    weights, _ = create_sampling_log_weights(df)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


def create_sampling_log_weights_v2(df, mu=0.5):
    label_counts = get_label_counts(df)
    label_weights = {}
    total = np.sum(list(label_counts.values()))
    for label, count in label_counts.items():
        score = math.log(mu * total / float(count))
        label_weights[label] = score if score > 1.0 else 1.0
    weights = []
    for i in range(df.shape[0]):
        ts = df.iloc[i]['Target'].split(' ')
        ws = [label_weights[t] for t in ts]
        w = max(ws)
        weights.append(w)
    return weights, label_weights


def create_weighted_random_sampler_v2(df):
    weights, _ = create_sampling_log_weights_v2(df)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    return sampler


def create_lr_scheduler(optimizer, patience=1, factor=0.5):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=patience, mode='max', factor=factor, verbose=True)
    return scheduler


def create_cosine_annealing_lr_scheduler(optimizer, batch_size, epoch_size,
                                         restart_period=5, t_mult=1.0):
    scheduler = CosineLRWithRestarts(optimizer, batch_size, epoch_size,
                                     restart_period=restart_period, t_mult=t_mult)
    return scheduler


def get_label_counts(df=None):
    if df is None:
        df = get_train_df()

    labels = {str(i): 0 for i in range(n_class)}
    for target in df['Target']:
        ts = target.split(' ')
        for t in ts:
            labels[t] += 1

    return labels


def read_ex_data_source_list():
    df = pd.read_csv(str(ex_csv))
    return df


def download_image(ex_id, color):
    """
    https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/69984
    """
    sp = ex_id.split('_', 1)
    img_path = f'{sp[0]}/{sp[1]}_{color}.jpg'
    img_url = ex_image_base_url + img_path

    # Get the raw response from the url
    r = requests.get(img_url, allow_redirects=True, stream=True)
    r.raw.decode_content = True

    # Use PIL to resize the image and to convert it to L
    # (8-bit pixels, black and white)
    im = Image.open(r.raw)
    im = im.resize((512, 512), Image.LANCZOS)

    return im


def save_ex_image(ex_id):
    try:
        for i, color in enumerate(['red', 'green', 'blue', 'yellow']):
            im = np.array(download_image(ex_id, color))
            if color == 'yellow':
                gray_im = (im[:, :, 0] + im[:, :, 1]) / 2
            else:
                gray_im = im[:, :, i]
            gray_im = gray_im.astype(np.uint8)
            gray_im = Image.fromarray(gray_im)
            gray_im.save(ex_image_dir / f'{ex_id}_{color}.png', 'PNG')
    except Exception as e:
        print(f'Failed: {e}')


def save_ex_images(pid, ex_ids):
    for i, ex_id in enumerate(ex_ids):
        save_ex_image(ex_id)


def get_ex_ids_to_save():
    ex_ids = []
    df = read_ex_data_source_list()
    for i in range(df.shape[0]):
        row = df.iloc[i]
        ts = set(row['Target'].split(' '))
        if ts & weak_classes:
            ex_ids.append(row['Id'])
    return ex_ids


def download_ex_data(ex_ids):
    process_num = 2
    list_len = len(ex_ids)

    pool = Pool(process_num)
    for i in range(process_num):
        start = int(i * list_len / process_num)
        end = int((i + 1) * list_len / process_num)
        process_ids = ex_ids[start:end]
        pool.apply_async(save_ex_images, args=(str(i), process_ids))

    print('Waiting ...')
    pool.close()
    pool.join()
    print('Done')


def get_enhanced_train_df():
    df = pd.read_csv(enhanced_train_csv)
    return df


def create_enhanced_train_csv():
    train_df = get_train_df()

    ex_df = read_ex_data_source_list()
    available_ex_ids = get_ex_ids_to_save()
    available_ex_df = ex_df[ex_df['Id'].isin(available_ex_ids)]

    train_df['Source'] = ['train' for _ in range(train_df.shape[0])]
    available_ex_df['Source'] = ['ex' for _ in range(available_ex_df.shape[0])]

    enhanced_train_df = pd.concat([train_df, available_ex_df])
    enhanced_train_df.to_csv(str(enhanced_train_csv), index=False)

    print(f'Save: {enhanced_train_csv}')


def inverted_image_list(df):
    inverted = {str(i): set() for i in range(n_class)}
    for i in range(df.shape[0]):
        row = df.iloc[i]
        target = row['Target']
        ts = target.split(' ')
        image_id = row['Id']
        for t in ts:
            inverted[t].add(image_id)

    return inverted


def create_undersampled_enhanced_train_csv(base_num=1200):
    enhanced_train_df = get_enhanced_train_df()

    label_counts = get_label_counts(enhanced_train_df)
    few_order_labels = sorted(label_counts, key=lambda k: label_counts[k])

    inverted_index = inverted_image_list(enhanced_train_df)

    sampled_label_counts = {label: 0 for label in label_counts.keys()}
    samples = set()
    for label in progress_bar(few_order_labels):
        candidates = inverted_index[label]

        if len(candidates) <= base_num:
            samples = samples | candidates
            for c in candidates:
                for lab in [str(i) for i in range(n_class) if c in inverted_index[str(i)]]:
                    sampled_label_counts[lab] += 1
            continue

        n_shortage = base_num - sampled_label_counts[label]
        if n_shortage <= 0:
            continue

        random.seed(0)
        subset = set(random.sample(candidates, k=n_shortage))
        samples = samples | subset
        for c in subset:
            for lab in [str(i) for i in range(n_class) if c in inverted_index[str(i)]]:
                sampled_label_counts[lab] += 1

    undersampled_enhanced_train_df = enhanced_train_df[enhanced_train_df['Id'].isin(samples)]
    undersampled_enhanced_train_df.to_csv(str(undersampled_enhanced_train_csv), index=False)

    print(f'Save: {undersampled_enhanced_train_csv}')


def get_undersampled_enhanced_train_df():
    df = pd.read_csv(undersampled_enhanced_train_csv)
    return df


def _mls_undersampled_enhanced_kfold_dfs():
    df = get_undersampled_enhanced_train_df()
    label_mat = multilabel_binary_representation(df, sparse=False)

    kf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for train_index, val_index in kf.split(df.index.values, label_mat):
        fold_train_df = df.iloc[train_index]
        fold_val_df = df.iloc[val_index]
        yield fold_train_df, fold_val_df


def generate_mls_undersampled_enhanced_kfold_cv5_list():
    for i, (train_df, val_df) in enumerate(_mls_undersampled_enhanced_kfold_dfs()):
        train_listfile = str(mls_undersampled_enhanced_kfold_cv5_list_dir / f'train_cv{i}.csv')
        val_listfile = str(mls_undersampled_enhanced_kfold_cv5_list_dir / f'val_cv{i}.csv')
        train_df.to_csv(train_listfile, index=False)
        val_df.to_csv(val_listfile, index=False)
        print(f'Generated: {train_listfile}')
        print(f'Generated: {val_listfile}')


def get_mls_undersampled_enhanced_train_val_df_fold(k):
    train_listfile = str(mls_undersampled_enhanced_kfold_cv5_list_dir / f'train_cv{k}.csv')
    val_listfile = str(mls_undersampled_enhanced_kfold_cv5_list_dir / f'val_cv{k}.csv')
    train_df = pd.read_csv(str(train_listfile))
    val_df = pd.read_csv(str(val_listfile))
    return train_df, val_df