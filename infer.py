import torch
import gc
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from data import *
from dataset import HPADataset, HPATestDataset


def evaluate(model, loader, **kwargs):
    """F1 score evaluation
    """
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in progress_bar(loader):
            pred = predict(model, data, **kwargs)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    score = f1_score(y_true, y_pred, average='macro')

    return score


def predict(model, x, device=None,
            use_sigmoid=True, threshold=0.5,
            with_tta=False):
    """
    Returns
    -------
    nd.array
        {0, 1} array of prediction
    """
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)

    x = x.to(device)
    logit = model(x)

    if with_tta:
        logit += model(x.flip(3))
        logit = 0.5 * logit

    if use_sigmoid:
        pred = torch.sigmoid(logit)
    else:
        pred = logit

    if threshold is None:
        return pred.cpu().numpy().astype(np.float32)

    if isinstance(threshold, list):
        # Multiple thresholds
        pred = pred.cpu().numpy()
        pred = (pred > threshold).astype(np.float32)
        return pred

    pred = (pred > threshold).cpu().numpy()
    return pred


def compute_best_thresholds(model, loader, average='macro', **kwargs):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in progress_bar(loader):
            kwargs['threshold'] = None
            pred = predict(model, data, **kwargs)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    best_thresholds = []
    for class_index in range(n_class):
        best_score = 0.0
        best_threshold = 0.0
        for th in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9):
            score = f1_score(
                y_true[:, class_index],
                (y_pred[:, class_index] > th).astype(np.float32),
                average=average)
            print(f'Class: {class_index}, th: {th}, score: {score}')
            if best_score < score:
                best_score = score
                best_threshold = th
        print(f'Class: {class_index}, Best threshold: {best_threshold}, Best score: {best_score}')
        print('-----')
        best_thresholds.append(best_threshold)

    best_f1 = f1_score(y_true, (y_pred > best_thresholds).astype(np.float32), average='macro')
    default_f1 = f1_score(y_true, (y_pred > 0.5).astype(np.float32), average='macro')

    print(f'Best F1: {best_f1}')
    print(f'Default F1: {default_f1}')

    return best_thresholds


def show_classification_report(model, cv=0, device=None, with_tta=False, use_adaptive_thresholds=True):
    model.eval()

    _, val_df = get_multilabel_stratified_train_val_df_fold(cv)
    val_image_db = open_images_h5_file()
    val_dataset = HPADataset(val_df, size=(512, 512), image_db=val_image_db, use_augmentation=False)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Get thresholds
    if use_adaptive_thresholds:
        thresholds = compute_best_thresholds(model, val_iter, average='binary', device=device,
                                             use_sigmoid=True, threshold=None, with_tta=with_tta)
    else:
        thresholds = 0.5
    print(f'Thresholds: {thresholds}')

    gc.collect()
    torch.cuda.empty_cache()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, target in progress_bar(val_iter):
            pred = predict(model, x, device=device, use_sigmoid=True,
                           threshold=thresholds, with_tta=with_tta)

            y_true.append(target.cpu().numpy())
            y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    for index, label in name_label_dict.items():
        print('-----')
        print(f'Label[{index}]: {label}')
        print(f'    - Accuracy: {accuracy_score(y_true[:, index], y_pred[:, index])}')
        print(f'    - F1: {f1_score(y_true[:, index], y_pred[:, index])}')
        print('-----')


def submission_pipeline(model, name, cv=0, device=None, with_tta=False, use_adaptive_thresholds=True):
    model.eval()

    _, val_df = get_multilabel_stratified_train_val_df_fold(cv)
    val_image_db = open_images_h5_file()
    val_dataset = HPADataset(val_df, size=(512, 512), image_db=val_image_db, use_augmentation=False)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

    df = get_test_df()
    image_db = open_test_images_h5_file()
    dataset = HPATestDataset(df, size=(512, 512), image_db=image_db)
    test_iter = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Get thresholds
    if use_adaptive_thresholds:
        thresholds = compute_best_thresholds(model, val_iter, average='binary', device=device,
                                             use_sigmoid=True, threshold=None, with_tta=with_tta)
    else:
        thresholds = 0.5
    print(f'Thresholds: {thresholds}')

    # Prediction
    zero_prediction_strategy = 'reduce_threshold'
    zero_label_count = 0
    predicted_labels = []
    with torch.no_grad():
        for x in progress_bar(test_iter):
            pred = predict(model, x, device=device, use_sigmoid=True,
                           threshold=None, with_tta=with_tta)
            for p in pred:
                wheres = np.argwhere(p > thresholds)
                if len(wheres) == 0:
                    zero_label_count += 1

                    if zero_prediction_strategy == 'max':
                        max_label = p.argmax()
                        predicted_labels.append(str(max_label))
                    elif zero_prediction_strategy == 'reduce_threshold':
                        reduced_thresholds = thresholds
                        while len(wheres) == 0:
                            if isinstance(thresholds, list):
                                reduced_thresholds = [th / 2.0 for th in reduced_thresholds]
                            elif isinstance(thresholds, float):
                                reduced_thresholds = reduced_thresholds / 2.0
                            wheres = np.argwhere(p > reduced_thresholds)
                        predicted_labels.append(' '.join(map(str, wheres.flatten())))
                    else:
                        raise ValueError(zero_prediction_strategy)
                else:
                    predicted_labels.append(' '.join(map(str, wheres.flatten())))
    print(f'Zero predictions: {zero_label_count}')

    # Save csv
    df['Predicted'] = predicted_labels
    filepath = submission_dir / f'{name}.csv'
    df.to_csv(str(filepath), index=False)
    print(f'Save: {filepath}')