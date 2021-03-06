import torch
import copy
import gc
import numpy as np
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from data import *
from dataset import HPADataset, HPATestDataset, HPAEnhancedDataset
import model as M


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
            with_tta=False, prob_weights=None,
            heavy_tta=False):
    """
    Returns
    -------
    nd.array
        {0, 1} array of prediction
    """
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)

    if isinstance(model, list):
        models = model
    else:
        models = [model]

    ys = np.zeros((x.shape[0], n_class))
    for i, model in enumerate(models):
        model.eval()

        x = x.to(device)
        logit = model(x)

        if use_sigmoid:
            y = torch.sigmoid(logit)
        else:
            y = logit

        y = y.cpu().numpy().astype(np.float32)

        if with_tta:
            if heavy_tta:
                logit = model(x.flip(3))
                if use_sigmoid:
                    y_tta = torch.sigmoid(logit)
                else:
                    y_tta = logit
                y += y_tta.cpu().numpy().astype(np.float32)

                logit = model(x.flip(2))
                if use_sigmoid:
                    y_tta = torch.sigmoid(logit)
                else:
                    y_tta = logit
                y += y_tta.cpu().numpy().astype(np.float32)

                logit = model(x.flip(2).flip(3))
                if use_sigmoid:
                    y_tta = torch.sigmoid(logit)
                else:
                    y_tta = logit
                y += y_tta.cpu().numpy().astype(np.float32)

                y = 0.25 * y
            else:
                logit = model(x.flip(3))

                if use_sigmoid:
                    y_tta = torch.sigmoid(logit)
                else:
                    y_tta = logit

                y += y_tta.cpu().numpy().astype(np.float32)
                y = 0.5 * y
        
        # Accumurate to ensemble
        if prob_weights is None:
            ys += y
        else:
            ys += prob_weights[i] * y

    if len(models) == 1:
        prob = ys
    else:
        if prob_weights is None:
            prob = ys / len(models)
        else:
            prob = ys / sum(prob_weights)

    if threshold is None:
        # Return probability instead of {0, 1}
        return prob

    if isinstance(threshold, list):
        # Multiple thresholds
        pred = (prob > threshold).astype(np.float32)
        return pred

    pred = (prob > threshold).astype(np.float32)
    return pred


def role_predict(role_models, x, device=None,
                 use_sigmoid=True, threshold=0.5,
                 with_tta=False):
    """
    Args
    ----
    role_models : dict
        {tuple[int]: model}

    Returns
    -------
    nd.array
        {0, 1} array of prediction
    """
    if x.ndimension() == 3:
        x = x.expand(1, *x.shape)

    ys = np.zeros((x.shape[0], n_class))
    for role_labels, model in role_models.items():
        model.eval()

        x = x.to(device)
        logit = model(x)

        if use_sigmoid:
            y = torch.sigmoid(logit)
        else:
            y = logit

        y = y.cpu().numpy().astype(np.float32)

        if with_tta:
            logit = model(x.flip(3))

            if use_sigmoid:
                y_tta = torch.sigmoid(logit)
            else:
                y_tta = logit

            y += y_tta.cpu().numpy().astype(np.float32)
            y = 0.5 * y
        
        ys[:, role_labels] = y[:, role_labels]

    prob = ys

    if threshold is None:
        # Return probability instead of {0, 1}
        return prob

    if isinstance(threshold, list):
        # Multiple thresholds
        pred = (prob > threshold).astype(np.float32)
        return pred

    pred = (prob > threshold).astype(np.float32)
    return pred


def compute_best_thresholds(model, loader, average='macro', **kwargs):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    use_role_prediction = isinstance(model, dict)
    print(f'Use role prediction: {use_role_prediction}')

    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in progress_bar(loader):
            kwargs['threshold'] = None

            if use_role_prediction:
                pred = role_predict(model, data, **kwargs)
            else:
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


def compute_best_thresholds_ensemble(models, cvs, **kwargs):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    image_db = open_images_h5_file()
    ex_image_db = open_ex_images_h5_file()
    ex_image_full_db = open_ex_images_full_h5_file()

    y_true = []
    y_pred = []

    for cv, model in zip(cvs, models):
        print(f'Load val_df MLS Enhanced full CV={cv}')
        _, val_df = get_mls_enhanced_full_train_val_df_fold(cv)

        val_dataset = HPAEnhancedDataset(val_df, size=(512, 512), image_db=image_db, ex_image_db=ex_image_db,
                                         ex_image_full_db=ex_image_full_db, use_augmentation=False)
        val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

        with torch.no_grad():
            for data, target in progress_bar(val_iter):
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
                average='binary')
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


def compute_best_thresholds_ensemble_v3(models, cvs, **kwargs):
    """
    """
    gc.collect()
    torch.cuda.empty_cache()

    image_db = open_images_h5_file()
    ex_image_db = open_ex_images_h5_file()
    ex_image_full_db = open_ex_images_full_h5_file()

    y_true = []
    y_pred = []

    for cv, model in zip(cvs, models):
        print(f'Load val_df MLS Enhanced full CV={cv}')
        _, val_df = get_mls_enhanced_full_train_val_df_fold(cv)

        val_dataset = HPAEnhancedDataset(val_df, size=(512, 512), image_db=image_db, ex_image_db=ex_image_db,
                                         ex_image_full_db=ex_image_full_db, use_augmentation=False)
        val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

        with torch.no_grad():
            for data, target in progress_bar(val_iter):
                kwargs['threshold'] = None
                pred = predict(model, data, **kwargs)

                y_true.append(target.cpu().numpy())
                y_pred.append(pred)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    ths = list(np.arange(0.01, 1.0, 0.01).astype(np.float32))
    best_thresholds = [0.5 for _ in range(n_class)] # Default 0.5
    for class_index in range(n_class):
        best_score = 0.0
        best_threshold_for_class = 0.0

        for th in ths:
            thresholds = [th if i == class_index else best_thresholds[i]
                          for i in range(n_class)]
            score = f1_score(y_true, (y_pred > thresholds).astype(np.float32), average='macro')
            if best_score < score:
                best_score = score
                best_threshold_for_class = th

        best_thresholds[class_index] = best_threshold_for_class

        print(f'Class: {class_index}, Best threshold for class: {best_threshold_for_class}, '
              f'Best F1 score: {best_score}, Best threshold: {best_thresholds}')
        print('-----')

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


def submission_pipeline(model, name, cv=0, device=None, with_tta=False,
                        use_adaptive_thresholds=True, fixed_threshold=0.5,
                        use_mls_v2=False, use_mls_us_enh=False,
                        use_mls_enh=False, use_mls_enh_full=False,
                        prob_weights=None,
                        zero_prediction_strategy='reduce_threshold',
                        heavy_tta=False):
    use_role_prediction = isinstance(model, dict)
    print(f'Use role prediction: {use_role_prediction}')

    if prob_weights is not None:
        print(f'Use prob_weights: {prob_weights}')

    print(f'Zero prediction strategy: {zero_prediction_strategy}')
    print(f'Heavy TTA: {heavy_tta}')

    if use_mls_v2:
        print('Load val_df MLS v2')
        _, val_df = get_multilabel_stratified_train_val_df_fold_v2(cv)
    elif use_mls_us_enh:
        print('Load val_df MLS Undersampled Enhanced')
        _, val_df = get_mls_undersampled_enhanced_train_val_df_fold(cv)
    elif use_mls_enh:
        print('Load val_df MLS Enhanced')
        _, val_df = get_mls_enhanced_train_val_df_fold(cv)
    elif use_mls_enh_full:
        print('Load val_df MLS Enhanced full')
        _, val_df = get_mls_enhanced_full_train_val_df_fold(cv)
    else:
        _, val_df = get_multilabel_stratified_train_val_df_fold(cv)

    image_db = open_images_h5_file()
    ex_image_db = open_ex_images_h5_file()
    ex_image_full_db = open_ex_images_full_h5_file()
    val_dataset = HPAEnhancedDataset(val_df, size=(512, 512), image_db=image_db, ex_image_db=ex_image_db,
                                     ex_image_full_db=ex_image_full_db, use_augmentation=False)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

    df = get_test_df()
    test_image_db = open_test_images_h5_file()
    dataset = HPATestDataset(df, size=(512, 512), image_db=test_image_db)
    test_iter = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True)

    # Get thresholds
    if use_adaptive_thresholds:
        assert prob_weights is None, 'TODO: Implement'
        thresholds = compute_best_thresholds(model, val_iter, average='binary', device=device,
                                             use_sigmoid=True, threshold=None, with_tta=with_tta)
    else:
        thresholds = fixed_threshold
    print(f'Thresholds: {thresholds}')

    # Prediction
    #zero_prediction_strategy = 'reduce_threshold'
    zero_label_count = 0
    predicted_labels = []
    with torch.no_grad():
        for x in progress_bar(test_iter):
            if use_role_prediction:
                pred = role_predict(model, x, device=device, use_sigmoid=True,
                                    threshold=None, with_tta=with_tta)
            else:
                pred = predict(model, x, device=device, use_sigmoid=True,
                               threshold=None, with_tta=with_tta, prob_weights=prob_weights,
                               heavy_tta=heavy_tta)

            for p in pred:
                wheres = np.argwhere(p > thresholds)
                if len(wheres) == 0:
                    zero_label_count += 1

                    if zero_prediction_strategy == 'max':
                        max_label = p.argmax()
                        predicted_labels.append(str(max_label))
                    elif zero_prediction_strategy == 'reduce_threshold':
                        reduced_thresholds = copy.deepcopy(thresholds)
                        while len(wheres) == 0:
                            if isinstance(thresholds, list):
                                reduced_thresholds = [th / 2.0 for th in reduced_thresholds]
                            elif isinstance(thresholds, float):
                                reduced_thresholds = reduced_thresholds / 2.0
                            wheres = np.argwhere(p > reduced_thresholds)
                        predicted_labels.append(' '.join(map(str, wheres.flatten())))
                    elif zero_prediction_strategy == 'reduce_threshold2':
                        reduced_thresholds = thresholds
                        while len(wheres) == 0:
                            if isinstance(thresholds, list):
                                reduced_thresholds = [th - 0.01 for th in reduced_thresholds]
                            elif isinstance(thresholds, float):
                                reduced_thresholds = reduced_thresholds - 0.01
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
    return df


def load_models(model_filenames, device=None):
    models = []
    for filename in model_filenames:
        model = M.ResNet34()

        modelfile = str(model_dir / filename)
        weight = torch.load(modelfile)
        model.load_state_dict(weight)
        model.to(device)
        model.eval()

        models.append(model)
        print(f'Loaded: {modelfile}')
    return models


def submission_pipeline_ensemble(models, name, cvs, device=None, with_tta=False,
                                 use_mls_enh_full=False):
    assert use_mls_enh_full

    image_db = open_images_h5_file()
    ex_image_db = open_ex_images_h5_file()
    ex_image_full_db = open_ex_images_full_h5_file()

    accumurated_best_thresholds = [0.0 for _ in range(n_class)]
    for cv, model in zip(cvs, models):
        print(f'Load val_df MLS Enhanced full CV={cv}')
        _, val_df = get_mls_enhanced_full_train_val_df_fold(cv)

        val_dataset = HPAEnhancedDataset(val_df, size=(512, 512), image_db=image_db, ex_image_db=ex_image_db,
                                         ex_image_full_db=ex_image_full_db, use_augmentation=False)
        val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)

        print(f'Compute threshold for CV={cv} ...')
        thresholds = compute_best_thresholds(model, val_iter, average='binary', device=device,
                                             use_sigmoid=True, threshold=None, with_tta=with_tta)
        print(f'Best thresholds(CV={cv}): {thresholds}')
        
        accumurated_best_thresholds = [at + t for at, t in zip(accumurated_best_thresholds, thresholds)]

    best_ensemble_thresholds = [t / len(cvs) for t in accumurated_best_thresholds]
    print(f'Best ensemble thresholds: {best_ensemble_thresholds}')

    df = get_test_df()
    test_image_db = open_test_images_h5_file()
    dataset = HPATestDataset(df, size=(512, 512), image_db=test_image_db)
    test_iter = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Prediction
    zero_prediction_strategy = 'reduce_threshold'
    zero_label_count = 0
    predicted_labels = []
    with torch.no_grad():
        for x in progress_bar(test_iter):
            pred = predict(models, x, device=device, use_sigmoid=True,
                           threshold=None, with_tta=with_tta)

            for p in pred:
                wheres = np.argwhere(p > best_ensemble_thresholds)
                if len(wheres) == 0:
                    zero_label_count += 1

                    if zero_prediction_strategy == 'max':
                        max_label = p.argmax()
                        predicted_labels.append(str(max_label))
                    elif zero_prediction_strategy == 'reduce_threshold':
                        reduced_thresholds = best_ensemble_thresholds
                        while len(wheres) == 0:
                            if isinstance(best_ensemble_thresholds, list):
                                reduced_thresholds = [th / 2.0 for th in reduced_thresholds]
                            elif isinstance(best_ensemble_thresholds, float):
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
    return df


def submission_pipeline_ensemble_v2(models, name, cvs, device=None, with_tta=False,
                                    use_mls_enh_full=False, threshold_computing_v3=False,
                                    zero_prediction_strategy='reduce_threshold',
                                    precomputed_thresholds=None):
    assert use_mls_enh_full

    if precomputed_thresholds:
        print('Use precomputed thresholds as best ensemble thresholds')
        best_ensemble_thresholds = precomputed_thresholds
    elif threshold_computing_v3:
        print('Compute threshold (v3) ...')
        best_ensemble_thresholds = compute_best_thresholds_ensemble_v3(models, cvs, device=device,
                                                                       use_sigmoid=True, threshold=None,
                                                                       with_tta=with_tta)
    else:
        print(f'Compute threshold ...')
        best_ensemble_thresholds = compute_best_thresholds_ensemble(models, cvs, device=device,
                                                                    use_sigmoid=True, threshold=None,
                                                                    with_tta=with_tta)
    print(f'Best ensemble thresholds: {best_ensemble_thresholds}')

    df = get_test_df()
    test_image_db = open_test_images_h5_file()
    dataset = HPATestDataset(df, size=(512, 512), image_db=test_image_db)
    test_iter = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, pin_memory=True)

    # Prediction
    #zero_prediction_strategy = 'reduce_threshold'
    zero_label_count = 0
    predicted_labels = []
    with torch.no_grad():
        for x in progress_bar(test_iter):
            pred = predict(models, x, device=device, use_sigmoid=True,
                           threshold=None, with_tta=with_tta)

            for p in pred:
                wheres = np.argwhere(p > best_ensemble_thresholds)
                if len(wheres) == 0:
                    zero_label_count += 1

                    if zero_prediction_strategy == 'max':
                        max_label = p.argmax()
                        predicted_labels.append(str(max_label))
                    elif zero_prediction_strategy == 'reduce_threshold':
                        reduced_thresholds = copy.deepcopy(best_ensemble_thresholds)
                        while len(wheres) == 0:
                            if isinstance(best_ensemble_thresholds, list):
                                reduced_thresholds = [th / 2.0 for th in reduced_thresholds]
                            elif isinstance(best_ensemble_thresholds, float):
                                reduced_thresholds = reduced_thresholds / 2.0
                            wheres = np.argwhere(p > reduced_thresholds)
                        predicted_labels.append(' '.join(map(str, wheres.flatten())))
                    elif zero_prediction_strategy == 'reduce_threshold3':
                        reduced_thresholds = copy.deepcopy(best_ensemble_thresholds)
                        while len(wheres) == 0:
                            if isinstance(best_ensemble_thresholds, list):
                                reduced_thresholds = [th - 0.05 for th in reduced_thresholds]
                            elif isinstance(best_ensemble_thresholds, float):
                                reduced_thresholds = reduced_thresholds - 0.05
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
    return df


final_thresholds = [0.43, 0.24, 0.34, 0.35, 0.33, 0.35, 0.28, 0.41, 0.55, 0.42, 0.36, 0.34, 0.36,
                    0.22, 0.29, 0.44, 0.15, 0.41, 0.21, 0.26, 0.35, 0.36, 0.23, 0.33, 0.43, 0.42, 0.27, 0.28]